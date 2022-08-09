from tokenizers.trainers import WordPieceTrainer
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenization import convert_to_unicode
import json
import pandas

corpus = pandas.read_csv('data/corpus.csv')

kor_corpus = corpus['원문']
with open('data/korean.txt', 'w', encoding='utf-8') as f:
    for line in kor_corpus:
        f.write(line + '\n')

eng_corpus = corpus['번역문']
with open('data/english.txt', 'w', encoding='utf-8') as f:
    for line in eng_corpus:
        f.write(line + '\n')

kor_tokenizer = Tokenizer(WordPiece())
kor_trainer = WordPieceTrainer(
    vocab_size=60000,
    special_tokens=['[PAD]', '[SOS]', '[EOS]', '[UNK]', '[MASK]', '[CLS]', '[SEP]', '<usr>', '<sys>', '<unused0>',
                    '<unused1>', '<unused2>', '<unused3>', '<unused4>', '<unused5>', '<unused6>', '<unused7>',
                    '<unused8>', '<unused9>', '<unused10>', '<unused11>', '<unused12>', '<unused13>', '<unused14>',
                    '<unused15>', '<unused16>', '<unused17>', '<unused18>', '<unused19>', '<unused20>', '<unused21>',
                    '<unused22>', '<unused23>', '<unused24>', '<unused25>', '<unused26>', '<unused27>', '<unused28>',
                    '<unused29>', '<unused30>', '<unused31>', '<unused32>', '<unused33>', '<unused34>', '<unused35>',
                    '<unused36>', '<unused37>', '<unused38>', '<unused39>', '<unused40>', '<unused41>', '<unused42>',
                    '<unused43>', '<unused44>', '<unused45>', '<unused46>', '<unused47>', '<unused48>', '<unused49>',
                    '<unused50>', '<unused51>', '<unused52>', '<unused53>', '<unused54>', '<unused55>', '<unused56>',
                    '<unused57>', '<unused58>', '<unused59>', '<unused60>', '<unused61>', '<unused62>', '<unused63>',
                    '<unused64>', '<unused65>', '<unused66>', '<unused67>', '<unused68>', '<unused69>', '<unused70>',
                    '<unused71>', '<unused72>', '<unused73>', '<unused74>', '<unused75>', '<unused76>', '<unused77>',
                    '<unused78>', '<unused79>', '<unused80>', '<unused81>', '<unused82>', '<unused83>', '<unused84>',
                    '<unused85>', '<unused86>', '<unused87>', '<unused88>', '<unused89>', '<unused90>', '<unused91>',
                    '<unused92>', '<unused93>', '<unused94>'],
    limit_alphabet=10000
)
kor_tokenizer.train(files=[f"data/korean.txt"], trainer=kor_trainer)
kor_tokenizer.save(f"vocab/korean.txt")

eng_tokenizer = Tokenizer(WordPiece())
eng_trainer = WordPieceTrainer(
    vocab_size=40000,
    special_tokens=['[PAD]', '[SOS]', '[EOS]', '[UNK]', '[MASK]', '[CLS]', '[SEP]', '<usr>', '<sys>', '<unused0>',
                    '<unused1>', '<unused2>', '<unused3>', '<unused4>', '<unused5>', '<unused6>', '<unused7>',
                    '<unused8>', '<unused9>', '<unused10>', '<unused11>', '<unused12>', '<unused13>', '<unused14>',
                    '<unused15>', '<unused16>', '<unused17>', '<unused18>', '<unused19>', '<unused20>', '<unused21>',
                    '<unused22>', '<unused23>', '<unused24>', '<unused25>', '<unused26>', '<unused27>', '<unused28>',
                    '<unused29>', '<unused30>', '<unused31>', '<unused32>', '<unused33>', '<unused34>', '<unused35>',
                    '<unused36>', '<unused37>', '<unused38>', '<unused39>', '<unused40>', '<unused41>', '<unused42>',
                    '<unused43>', '<unused44>', '<unused45>', '<unused46>', '<unused47>', '<unused48>', '<unused49>',
                    '<unused50>', '<unused51>', '<unused52>', '<unused53>', '<unused54>', '<unused55>', '<unused56>',
                    '<unused57>', '<unused58>', '<unused59>', '<unused60>', '<unused61>', '<unused62>', '<unused63>',
                    '<unused64>', '<unused65>', '<unused66>', '<unused67>', '<unused68>', '<unused69>', '<unused70>',
                    '<unused71>', '<unused72>', '<unused73>', '<unused74>', '<unused75>', '<unused76>', '<unused77>',
                    '<unused78>', '<unused79>', '<unused80>', '<unused81>', '<unused82>', '<unused83>', '<unused84>',
                    '<unused85>', '<unused86>', '<unused87>', '<unused88>', '<unused89>', '<unused90>', '<unused91>',
                    '<unused92>', '<unused93>', '<unused94>'],
    limit_alphabet=10000
)
eng_tokenizer.train(files=[f"data/english.txt"], trainer=eng_trainer)
eng_tokenizer.save(f"vocab/english.txt")

with open(f"vocab/korean.txt", "rb") as f:
    kor_vocab = json.load(f)
    kor_vocab = kor_vocab['model']['vocab']
    kor_vocab_size = len(kor_vocab) - 1
    with open(f"vocab/kor_vocab.txt", "w", encoding='utf-8') as ff:
        for key in kor_vocab:
            key = convert_to_unicode(key)
            key = key.replace('\n', '')
            key = key.replace('\r', '')
            if key.replace(' ', '') != '' and len(key) > 0:
                ff.write(key + '\n')

with open(f"vocab/english.txt", "rb") as f:
    eng_vocab = json.load(f)
    eng_vocab = eng_vocab['model']['vocab']
    eng_vocab_size = len(eng_vocab) - 1
    with open(f"vocab/eng_vocab.txt", "w", encoding='utf-8') as ff:
        for key in eng_vocab:
            key = convert_to_unicode(key)
            key = key.replace('\n', '')
            key = key.replace('\r', '')
            if key.replace(' ', '') != '' and len(key) > 0:
                ff.write(key + '\n')

with open('config.json', 'r') as f:
    config = json.load(f)
    config['kor_vocab_path'] = f"vocab/kor_vocab.txt"
    config['eng_vocab_path'] = f"vocab/eng_vocab.txt"
    config['kor_vocab_length'] = kor_vocab_size
    config['eng_vocab_length'] = eng_vocab_size
    with open('config.json', 'w') as w:
        json.dump(config, w, indent=4)