import torch
import tokenization
import argparse


def test(model, test_sentence, mode):
    if mode == "eng2kor":
        src_tokenizer = tokenization.FullTokenizer(
            vocab_file='vocab/eng_vocab.txt', do_lower_case=False)
        tgt_tokenizer = tokenization.FullTokenizer(
            vocab_file='vocab/kor_vocab.txt', do_lower_case=False)
    elif mode == "kor2eng":
        src_tokenizer = tokenization.FullTokenizer(
            vocab_file='vocab/kor_vocab.txt', do_lower_case=False)
        tgt_tokenizer = tokenization.FullTokenizer(
            vocab_file='vocab/eng_vocab.txt', do_lower_case=False)
    else:
        raise Exception("mode should be either 'eng2kor' or 'kor2eng'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer = torch.load(model, map_location=device)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/transformer_0.pth', help='model path')
    parser.add_argument('--test_sentence', type=str, default='Neural Machine Translation (NMT) [41, 2] has recently been introduced as a promising approach with the potential of addressing many shortcomings of traditional machine translation systems.', help='test sentence')
    parser.add_argument('--mode', type=str, default='eng2kor', help='mode')
    args = parser.parse_args()
    test(args.model, args.test_sentence, args.mode)
