import six
import json
import torch
import tokenization
import datasets

from transformer.model import Translation

def train():

    config = json.load(open('config.json'))
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_tokenizer = tokenization.FullTokenizer(
        vocab_file='vocab/kor_vocab.txt', do_lower_case=False)
    tgt_tokenizer = tokenization.FullTokenizer(
        vocab_file='vocab/eng_vocab.txt', do_lower_case=False)

    src_pad_index = src_tokenizer.convert_tokens_to_ids([config['pad_word']])[0]
    tgt_pad_index = tgt_tokenizer.convert_tokens_to_ids([config['pad_word']])[0]

    transformer = Translation(
        src_vocab_size=config['kor_vocab_length'],
        tgt_vocab_size=config['eng_vocab_length'],
        d_model=config['d_model'], d_ff=config['d_ff'],
        d_k=config['d_k'], d_v=config['d_v'], n_heads=config['num_heads'], 
        n_layers=config['num_layers'], src_pad_index=src_pad_index,
        tgt_pad_index=src_pad_index, device=device).to(device)

    dataset = datasets.TranslationDataset(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_length=40,
        tgt_length=40,
        file_root='data/corpus.csv',
        src_column='원문',
        tgt_column='번역문')

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, num_workers=4,
        batch_size=64, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)

    test_sentences = [
        '본 발명에서 자성의 세기로는 50 ~ 15000 G 정도를 가지는 영구자석이라면 적용 가능하다',
        '이 경우의 유기 액체란 적주식 분해 가스의 원료를 가리키는데, 이미 널리 사용되고 있는 CH3OH가 주체로 생각된다.',
        '유리화된 영역에는 콜라겐 배경의 형질 세포, 림프구 및 비정형 방추 세포가 포함된다.',
        '이러한 규칙을 지키면서 사용할 수 있는 패턴의 종류는 389,112개가 존재한다',
        '내부 카메라 모듈(126c)는 탑승자에 대한 이미지를 획득할 수 있다',
        '서브 픽셀을 구성하는 박막 트랜지스터들은 p 타입으로 구현되거나 또는, n 타입으로 구현될 수 있다.']

    for epoch in range(4):
        total_loss = 0
        for step, (enc_inputs, dec_inputs, target_batch) in enumerate(dataloader):
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = transformer(enc_inputs, dec_inputs)
            loss = criterion(
                dec_logits.view(-1, dec_logits.size(-1)),
                target_batch.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print('----------------------')
            print(f'epoch : {epoch}')
            print(f'step  : {step}/{len(dataloader)}')
            print(f'loss  : {loss.item()}')
            print(f'lr    : {scheduler.get_last_lr()}')
        scheduler.step()
        print('----------------------')
        print(f'epoch : {epoch}')
        print(f'loss  : {total_loss / len(dataloader)}')
        print(f'lr    : {scheduler.get_last_lr()}')
        torch.save(transformer.state_dict(), f'models/transformer_{epoch}.pth')
        print(f'saved model to "models/transformer_{epoch}.pth"')

        for test_sentence in test_sentences:
            orig_text = test_sentence
            test_sentence = src_tokenizer.tokenize(test_sentence)
            test_sentence_ids = src_tokenizer.convert_tokens_to_ids(test_sentence)
            enc_token = torch.as_tensor([test_sentence_ids], dtype=torch.long)

            test_sentence_dec = ['[SOS]']
            test_sentence_dec = tgt_tokenizer.convert_tokens_to_ids(test_sentence_dec)
            eos_flag = tgt_tokenizer.convert_tokens_to_ids(['[EOS]'])

            while test_sentence_dec[-1] != eos_flag[0]:
                dec_input = torch.as_tensor([test_sentence_dec], dtype=torch.long)
                enc_token, dec_input = enc_token.to(device), dec_input.to(device)
                dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = transformer(enc_token, dec_input)
                predict = torch.argmax(dec_logits, axis=2)[:, -1].squeeze().detach().cpu().numpy()
                test_sentence_dec.append(int(predict))

            predict_text = ' '.join(tgt_tokenizer.convert_ids_to_tokens(test_sentence_dec))
            predict_text = predict_text.replace(" ##", "")
            predict_text = predict_text.replace("##", "")
            print(f'orig_text    : {orig_text}')
            print(f'predict_text : {predict_text}')
            print('-----------------')

if __name__ == '__main__':
    train()
