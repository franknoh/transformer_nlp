import torch
import tokenization
import datasets
import wandb
import time
from transformer.model import Translation

wandb.init(project="transformer_nlp", entity="franknoh")

wandb.config = {
    "kor_vocab_length": 50000,
    "eng_vocab_length": 28998,
    "d_model": 768,
    "d_ff": 2048,
    "d_k": 64,
    "d_v": 64,
    "num_layers": 12,
    "num_heads": 8,
    "start_word": "[SOS]",
    "end_word": "[EOS]",
    "sep_word": "[SEP]",
    "cls_word": "[CLS]",
    "pad_word": "[PAD]",
    "mask_word": "[MASK]",
    "max_seq_length": 40,
    "num_train_epochs": 4,
    "batch_size": 64,
    "learning_rate": 0.00005,
    "optimizer_gamma": 0.1
}


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kor_tokenizer = tokenization.FullTokenizer(
        vocab_file='vocab/kor_vocab.txt', do_lower_case=False)
    eng_tokenizer = tokenization.FullTokenizer(
        vocab_file='vocab/eng_vocab.txt', do_lower_case=False)

    kor_pad_index = kor_tokenizer.convert_tokens_to_ids([wandb.config['pad_word']])[0]
    eng_pad_index = eng_tokenizer.convert_tokens_to_ids([wandb.config['pad_word']])[0]

    kor2eng_transformer = Translation(
        src_vocab_size=wandb.config['kor_vocab_length'],
        tgt_vocab_size=wandb.config['eng_vocab_length'],
        d_model=wandb.config['d_model'], d_ff=wandb.config['d_ff'],
        d_k=wandb.config['d_k'], d_v=wandb.config['d_v'], n_heads=wandb.config['num_heads'],
        n_layers=wandb.config['num_layers'], src_pad_index=kor_pad_index,
        tgt_pad_index=kor_pad_index, device=device).to(device)

    eng2kor_transformer = Translation(
        src_vocab_size=wandb.config['eng_vocab_length'],
        tgt_vocab_size=wandb.config['kor_vocab_length'],
        d_model=wandb.config['d_model'], d_ff=wandb.config['d_ff'],
        d_k=wandb.config['d_k'], d_v=wandb.config['d_v'], n_heads=wandb.config['num_heads'],
        n_layers=wandb.config['num_layers'], src_pad_index=eng_pad_index,
        tgt_pad_index=eng_pad_index, device=device).to(device)

    dataset = datasets.TranslationDataset(
        src_tokenizer=kor_tokenizer,
        tgt_tokenizer=eng_tokenizer,
        src_length=wandb.config['max_seq_length'],
        tgt_length=wandb.config['max_seq_length'],
        file_root='data/corpus.csv',
        src_column='원문',
        tgt_column='번역문')

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, num_workers=4,
        batch_size=wandb.config['batch_size'], shuffle=True)

    kor2eng_criterion = torch.nn.CrossEntropyLoss()
    kor2eng_optimizer = torch.optim.Adam(kor2eng_transformer.parameters(), lr=wandb.config['learning_rate'])
    kor2eng_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=kor2eng_optimizer,
                                                               gamma=wandb.config['optimizer_gamma'])

    eng2kor_criterion = torch.nn.CrossEntropyLoss()
    eng2kor_optimizer = torch.optim.Adam(eng2kor_transformer.parameters(), lr=wandb.config['learning_rate'])
    eng2kor_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=eng2kor_optimizer,
                                                               gamma=wandb.config['optimizer_gamma'])

    kor2eng_test_sentences = [
        '본 발명에서 자성의 세기로는 50 ~ 15000 G 정도를 가지는 영구자석이라면 적용 가능하다',
        '이 경우의 유기 액체란 적주식 분해 가스의 원료를 가리키는데, 이미 널리 사용되고 있는 CH3OH가 주체로 생각된다.',
        '유리화된 영역에는 콜라겐 배경의 형질 세포, 림프구 및 비정형 방추 세포가 포함된다.',
        '이러한 규칙을 지키면서 사용할 수 있는 패턴의 종류는 389,112개가 존재한다',
        '내부 카메라 모듈(126c)는 탑승자에 대한 이미지를 획득할 수 있다',
        '서브 픽셀을 구성하는 박막 트랜지스터들은 p 타입으로 구현되거나 또는, n 타입으로 구현될 수 있다.']

    eng2kor_test_sentences = [
        'Neural Machine Translation (NMT) [41, 2] has recently been introduced as a promising approach with the '
        'potential of addressing many shortcomings of traditional machine translation systems.',
        'The input sensing unit ISU may sense various types of inputs provided from the outside of the electronic '
        'device ED.',
        'While performing the task, it is possible to charge the electric energy by landing on the three-phase '
        'wireless charging station 200 if necessary. ',
        'Also, single-cell RNA sequencing allows detection of rare cancer cells and CTCs, elucidation of mechanism '
        'There is a consideration which appears at first sight to be opposed to the admission of our hypothesis with '
        'respect to compound substances.for drug resistance and prediction of prognosis of cancer patients.'
    ]

    for epoch in range(wandb.config['num_train_epochs']):
        kor2eng_total_loss = 0
        eng2kor_total_loss = 0
        wandb_loss = []
        total_time = time.time()
        for step, (kor_a_inputs, eng_a_inputs, kor2eng_target_batch, kor_b_inputs, eng_b_inputs, eng2kor_target_batch) in enumerate(dataloader):
            start = time.time()

            kor_a_inputs = kor_a_inputs.to(device)
            eng_a_inputs = eng_a_inputs.to(device)
            kor_b_inputs = kor_b_inputs.to(device)
            eng_b_inputs = eng_b_inputs.to(device)

            kor2eng_target_batch = kor2eng_target_batch.to(device)
            eng2kor_target_batch = eng2kor_target_batch.to(device)

            kor2eng_optimizer.zero_grad()
            dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = kor2eng_transformer(kor_a_inputs, eng_a_inputs)
            kor2eng_loss = kor2eng_criterion(
                dec_logits.view(-1, dec_logits.size(-1)),
                kor2eng_target_batch.contiguous().view(-1))
            kor2eng_loss.backward()
            kor2eng_optimizer.step()
            kor2eng_total_loss += kor2eng_loss.item()

            eng2kor_optimizer.zero_grad()
            dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = eng2kor_transformer(eng_b_inputs, kor_b_inputs)
            eng2kor_loss = eng2kor_criterion(
                dec_logits.view(-1, dec_logits.size(-1)),
                eng2kor_target_batch.contiguous().view(-1))
            eng2kor_loss.backward()
            eng2kor_optimizer.step()
            eng2kor_total_loss += eng2kor_loss.item()

            print('--------kor2eng--------')
            print(f'loss  : {kor2eng_loss.item()}')
            print(f'lr    : {kor2eng_scheduler.get_last_lr()}')
            print('--------eng2kor--------')
            print(f'loss  : {eng2kor_loss.item()}')
            print(f'lr    : {eng2kor_scheduler.get_last_lr()}')
            print(f'epoch : {epoch + 1}/{wandb.config["num_train_epochs"]}')
            print(f'step  : {step}/{len(dataloader)}')
            print(f'time  : {time.time() - start}sec')
            print(f'eta   : {(len(dataloader) - step) * (time.time() - start) / 60:.2f}min')
            wandb.log({'kor2eng_loss': kor2eng_loss, 'eng2kor_loss': eng2kor_loss,
                       'combined_loss': 2 / (1 / kor2eng_loss + 1 / eng2kor_loss)})
        kor2eng_scheduler.step()
        eng2kor_scheduler.step()
        print('----------------------')
        print(f'epoch         : {epoch + 1}/{wandb.config["num_train_epochs"]}')
        print(f'total time    : {(time.time() - total_time) / 3600:.2f}hr')
        print(f'kor2eng loss  : {kor2eng_total_loss / len(dataloader)}')
        print(f'kor2eng lr    : {kor2eng_scheduler.get_last_lr()}')
        print(f'eng2kor loss  : {eng2kor_total_loss / len(dataloader)}')
        print(f'eng2kor lr    : {eng2kor_scheduler.get_last_lr()}')
        torch.save(kor2eng_transformer.state_dict(), f'models/kor2eng_{epoch}.pth')
        torch.save(eng2kor_transformer.state_dict(), f'models/eng2kor_{epoch}.pth')
        print(f'saved model to "models/kor2eng_{epoch}.pth", "models/eng2kor_{epoch}.pth"')

        for test_sentence in kor2eng_test_sentences:
            orig_text = test_sentence
            test_sentence = kor_tokenizer.tokenize(test_sentence)
            test_sentence_ids = kor_tokenizer.convert_tokens_to_ids(test_sentence)
            enc_token = torch.as_tensor([test_sentence_ids], dtype=torch.long)

            test_sentence_dec = ['[SOS]']
            test_sentence_dec = eng_tokenizer.convert_tokens_to_ids(test_sentence_dec)
            eos_flag = eng_tokenizer.convert_tokens_to_ids(['[EOS]'])

            while test_sentence_dec[-1] != eos_flag[0]:
                dec_input = torch.as_tensor([test_sentence_dec], dtype=torch.long)
                enc_token, dec_input = enc_token.to(device), dec_input.to(device)
                dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = kor2eng_transformer(enc_token, dec_input)
                predict = torch.argmax(dec_logits, axis=2)[:, -1].squeeze().detach().cpu().numpy()
                test_sentence_dec.append(int(predict))

            predict_text = ' '.join(eng_tokenizer.convert_ids_to_tokens(test_sentence_dec))
            predict_text = predict_text.replace(" ##", "")
            predict_text = predict_text.replace("##", "")
            print(f'original_text    : {orig_text}')
            print(f'predict_text : {predict_text}')
            print('-----------------')

        for test_sentence in eng2kor_test_sentences:
            orig_text = test_sentence
            test_sentence = eng_tokenizer.tokenize(test_sentence)
            test_sentence_ids = eng_tokenizer.convert_tokens_to_ids(test_sentence)
            enc_token = torch.as_tensor([test_sentence_ids], dtype=torch.long)

            test_sentence_dec = ['[SOS]']
            test_sentence_dec = kor_tokenizer.convert_tokens_to_ids(test_sentence_dec)
            eos_flag = kor_tokenizer.convert_tokens_to_ids(['[EOS]'])

            while test_sentence_dec[-1] != eos_flag[0]:
                dec_input = torch.as_tensor([test_sentence_dec], dtype=torch.long)
                enc_token, dec_input = enc_token.to(device), dec_input.to(device)
                dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = eng2kor_transformer(enc_token, dec_input)
                predict = torch.argmax(dec_logits, axis=2)[:, -1].squeeze().detach().cpu().numpy()
                test_sentence_dec.append(int(predict))

            predict_text = ' '.join(kor_tokenizer.convert_ids_to_tokens(test_sentence_dec))
            predict_text = predict_text.replace(" ##", "")
            predict_text = predict_text.replace("##", "")
            print(f'original_text    : {orig_text}')
            print(f'predict_text : {predict_text}')
            print('-----------------')


if __name__ == '__main__':
    train()
