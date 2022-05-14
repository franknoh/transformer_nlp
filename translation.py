import torch
import tokenization
import argparse
import json
import time
from transformer.model import Translation

with open("config.json", "r") as f:
    config = json.load(f)


def test(model, test_sentence, mode, kor_vocab, eng_vocab):
    if mode == "eng2kor":
        src_tokenizer = tokenization.FullTokenizer(
            vocab_file=kor_vocab, do_lower_case=False)
        tgt_tokenizer = tokenization.FullTokenizer(
            vocab_file=eng_vocab, do_lower_case=False)
    elif mode == "kor2eng":
        src_tokenizer = tokenization.FullTokenizer(
            vocab_file=kor_vocab, do_lower_case=False)
        tgt_tokenizer = tokenization.FullTokenizer(
            vocab_file=eng_vocab, do_lower_case=False)
    else:
        raise Exception("mode should be either 'eng2kor' or 'kor2eng'")

    print("loading model...")

    src_pad_index = src_tokenizer.convert_tokens_to_ids([config['pad_word']])[0]
    tgt_pad_index = tgt_tokenizer.convert_tokens_to_ids([config['pad_word']])[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer = Translation(
        src_vocab_size=config['kor_vocab_length'],
        tgt_vocab_size=config['eng_vocab_length'],
        d_model=config['d_model'], d_ff=config['d_ff'],
        d_k=config['d_k'], d_v=config['d_v'], n_heads=config['num_heads'],
        n_layers=config['num_layers'], src_pad_index=src_pad_index,
        tgt_pad_index=tgt_pad_index, device=device).to(device)
    transformer.load_state_dict(torch.load(model, map_location=device))

    print("done")

    start_time = time.time()
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

    predict_tokens = tgt_tokenizer.convert_ids_to_tokens(test_sentence_dec)
    predict_text = ' '.join(predict_tokens[1:-1])
    predict_text = predict_text.replace(" ##", "")
    predict_text = predict_text.replace("##", "")
    print(f'orignal text : {orig_text}')
    print(f'predict text : {predict_text}')
    print(f'elapsed time : {time.time() - start_time}sec')
    print('-----------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/kor2eng.pth', help='model path')
    parser.add_argument('--test_sentence', type=str, default='각 전자석의 회전 방향이나 이동 방향은 반드시 일치시켜 둘 필요는 없다.',
                        help='test sentence')
    parser.add_argument('--mode', type=str, default='kor2eng', help='mode')
    parser.add_argument('--kor_vocab', type=str, default='vocab/korean_vocab.txt', help='korean vocab')
    parser.add_argument('--eng_vocab', type=str, default='vocab/english_vocab.txt', help='english vocab')
    args = parser.parse_args()
    test(args.model, args.test_sentence, args.mode, args.kor_vocab, args.eng_vocab)
