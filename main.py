import torch
import tokenization
import datasets
import wandb
import time
import json
import argparse
import utils
from math import sqrt
from transformer.model import Translation
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

def train(num_train_epochs, ckpt_path, dataset_path, learning_rate, batch_size, max_seq_length, optimizer_gamma):
    with open("config.json", "r") as f:
        config = json.load(f)
    if config['wandb_name'] is not None:
        wandb.init(project=config['wandb_id'], entity=config['wandb_name'])
        wandb.config = config
    wandb.config['learning_rate'] = learning_rate
    wandb.config['batch_size'] = batch_size
    wandb.config['max_seq_length'] = max_seq_length
    wandb.config['optimizer_gamma'] = optimizer_gamma

    kor_tokenizer = tokenization.FullTokenizer(
        vocab_file=wandb.config['kor_vocab_path'], do_lower_case=False)
    eng_tokenizer = tokenization.FullTokenizer(
        vocab_file=wandb.config['eng_vocab_path'], do_lower_case=False)

    kor_pad_index = kor_tokenizer.convert_tokens_to_ids([wandb.config['pad_word']])[0]
    eng_pad_index = eng_tokenizer.convert_tokens_to_ids([wandb.config['pad_word']])[0]

    kor2eng_transformer = Translation(
        src_vocab_size=wandb.config['kor_vocab_length'],
        tgt_vocab_size=wandb.config['eng_vocab_length'],
        d_model=wandb.config['d_model'], d_ff=wandb.config['d_ff'],
        d_k=wandb.config['d_k'], d_v=wandb.config['d_v'], n_heads=wandb.config['num_heads'],
        n_layers=wandb.config['num_layers'], src_pad_index=kor_pad_index,
        tgt_pad_index=eng_pad_index, device=device).to(device)

    eng2kor_transformer = Translation(
        src_vocab_size=wandb.config['eng_vocab_length'],
        tgt_vocab_size=wandb.config['kor_vocab_length'],
        d_model=wandb.config['d_model'], d_ff=wandb.config['d_ff'],
        d_k=wandb.config['d_k'], d_v=wandb.config['d_v'], n_heads=wandb.config['num_heads'],
        n_layers=wandb.config['num_layers'], src_pad_index=eng_pad_index,
        tgt_pad_index=kor_pad_index, device=device).to(device)

    if ckpt_path != '':
        kor2eng_transformer.load_state_dict(torch.load(f'{ckpt_path}/kor2eng.pth', map_location=device))
        eng2kor_transformer.load_state_dict(torch.load(f'{ckpt_path}/eng2kor.pth', map_location=device))
    kor2eng_dataset = datasets.TranslationDataset(
        src_tokenizer=kor_tokenizer,
        tgt_tokenizer=eng_tokenizer,
        src_length=wandb.config['max_seq_length'],
        tgt_length=wandb.config['max_seq_length'],
        file_root=dataset_path,
        src_column='원문',
        tgt_column='번역문')

    eng2kor_dataset = datasets.TranslationDataset(
        src_tokenizer=eng_tokenizer,
        tgt_tokenizer=kor_tokenizer,
        src_length=wandb.config['max_seq_length'],
        tgt_length=wandb.config['max_seq_length'],
        file_root=dataset_path,
        src_column='번역문',
        tgt_column='원문')

    kor2eng_dataloader = torch.utils.data.DataLoader(
        dataset=kor2eng_dataset, num_workers=2,
        batch_size=wandb.config['batch_size'], shuffle=True)

    eng2kor_dataloader = torch.utils.data.DataLoader(
        dataset=eng2kor_dataset, num_workers=2,
        batch_size=wandb.config['batch_size'], shuffle=True)

    kor2eng_criterion = torch.nn.CrossEntropyLoss()
    kor2eng_optimizer = torch.optim.Adam(kor2eng_transformer.parameters(), lr=wandb.config['learning_rate'])
    kor2eng_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=kor2eng_optimizer, gamma=wandb.config['optimizer_gamma'])

    eng2kor_criterion = torch.nn.CrossEntropyLoss()
    eng2kor_optimizer = torch.optim.Adam(eng2kor_transformer.parameters(), lr=wandb.config['learning_rate'])
    eng2kor_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=eng2kor_optimizer, gamma=wandb.config['optimizer_gamma'])

    kor2eng_optimizer, kor2eng_scheduler, kor2eng_criterion, kor2eng_dataloader, eng2kor_optimizer, eng2kor_scheduler, eng2kor_criterion, eng2kor_dataloader = accelerator.prepare(
        kor2eng_optimizer, kor2eng_scheduler, kor2eng_criterion, kor2eng_dataloader, eng2kor_optimizer,
        eng2kor_scheduler, eng2kor_criterion, eng2kor_dataloader
    )

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

    for epoch in range(num_train_epochs):
        kor2eng_total_loss = 0
        eng2kor_total_loss = 0
        wandb_loss = []
        total_time = time.time()
        for step, (enc_inputs, dec_inputs, target_batch) in enumerate(kor2eng_dataloader):
            start = time.time()
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            target_batch = target_batch.to(device)

            kor2eng_optimizer.zero_grad()
            dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = kor2eng_transformer(enc_inputs, dec_inputs)
            loss = kor2eng_criterion(
                dec_logits.view(-1, dec_logits.size(-1)),
                target_batch.contiguous().view(-1))
            accelerator.backward(loss)
            kor2eng_optimizer.step()

            kor2eng_total_loss += loss.item()

            print('--------kor2eng--------')
            print(f'epoch : {epoch+1}/{num_train_epochs}')
            print(f'step  : {step}/{len(kor2eng_dataloader)}')
            print(f'loss  : {loss.item()}')
            print(f'lr    : {kor2eng_scheduler.get_last_lr()}')
            print(f'time  : {utils.sec_to_time(time.time() - start)}')
            print(f'eta   : {(len(kor2eng_dataloader) - step)* (time.time() - start)/60:.2f}min')
            wandb_loss.append([loss.item()])
        kor2eng_scheduler.step()
        for step, (enc_inputs, dec_inputs, target_batch) in enumerate(eng2kor_dataloader):
            start = time.time()
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            target_batch = target_batch.to(device)

            eng2kor_optimizer.zero_grad()
            dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = eng2kor_transformer(enc_inputs, dec_inputs)
            loss = eng2kor_criterion(
                dec_logits.view(-1, dec_logits.size(-1)),
                target_batch.contiguous().view(-1))
            accelerator.backward(loss)
            eng2kor_optimizer.step()

            eng2kor_total_loss += loss.item()

            print('--------eng2kor--------')
            print(f'epoch : {epoch+1}/{num_train_epochs}')
            print(f'step  : {step}/{len(eng2kor_dataloader)}')
            print(f'loss  : {loss.item()}')
            print(f'lr    : {eng2kor_scheduler.get_last_lr()}')
            print(f'time  : {utils.sec_to_time(time.time() - start)}')
            print(f'eta   : {(len(eng2kor_dataloader) - step)* (time.time() - start)/60:.2f}min')
            wandb_loss[step].append(loss.item())
        eng2kor_scheduler.step()
        for loss in wandb_loss:
            wandb.log({'kor2eng_loss': loss[0], 'eng2kor_loss': loss[1], 'combined_loss': sqrt(loss[0]**2 + loss[1]**2)})

        print('----------------------')
        print(f'epoch         : {epoch+1}/{num_train_epochs}')
        print(f'total time    : {utils.sec_to_time(time.time() - total_time)}')
        print(f'kor2eng loss  : {kor2eng_total_loss / len(kor2eng_dataloader)}')
        print(f'kor2eng lr    : {kor2eng_scheduler.get_last_lr()}')
        print(f'eng2kor loss  : {eng2kor_total_loss / len(eng2kor_dataloader)}')
        print(f'eng2kor lr    : {eng2kor_scheduler.get_last_lr()}')
        accelerator.save(kor2eng_transformer.state_dict(), f'models/kor2eng_{epoch}.pth')
        accelerator.save(eng2kor_transformer.state_dict(), f'models/eng2kor_{epoch}.pth')
        accelerator.save(kor2eng_transformer.state_dict(), f'models/kor2eng.pth')
        accelerator.save(eng2kor_transformer.state_dict(), f'models/eng2kor.pth')
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_epochs', type=int, default=20, help='number of train epochs')
    parser.add_argument('--ckpt_path', type=str, default='', help='path to load model')
    parser.add_argument('--dataset_path', type=str, default='data/corpus.csv', help='path to load dataset')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--max_seq_length', type=int, default=120, help='max sequence length')
    parser.add_argument('--optimizer_gamma', type=float, default=0.9, help='optimizer gamma')
    args = parser.parse_args()
    train(args.num_train_epochs, args.ckpt_path, args.dataset_path, args.learning_rate, args.batch_size, args.max_seq_length, args.optimizer_gamma)