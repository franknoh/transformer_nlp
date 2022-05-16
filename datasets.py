import pandas
import torch


class TranslationDataset(torch.utils.data.Dataset):

    def __init__(self, src_tokenizer, tgt_tokenizer,
                 file_root, src_column, tgt_column,
                 src_length, tgt_length,
                 start_token='[SOS]', end_token='[EOS]',
                 padding_token='[PAD]'):

        self.dataframe = pandas.read_csv(file_root)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token
        self.src_column = src_column
        self.tgt_column = tgt_column
        self.src_length = src_length
        self.tgt_length = tgt_length

    def __getitem__(self, idx):
        src = self.dataframe.loc[idx, self.src_column]

        tokenized_src = self.src_tokenizer.tokenize(src)
        tokenized_src = tokenized_src[:self.src_length]
        while len(tokenized_src) < self.src_length:
            tokenized_src.append(self.padding_token)
        inp = self.src_tokenizer.convert_tokens_to_ids(tokenized_src)

        tgt_token = [self.start_token]
        tgt = self.dataframe.loc[idx, self.tgt_column]
        tgt = self.tgt_tokenizer.tokenize(tgt)
        tgt_token.extend(tgt)
        tgt_token.append(self.end_token)

        tgt_input = tgt_token[:-1]
        tgt_output = tgt_token[1:]

        tgt_input = tgt_input[:self.tgt_length]
        tgt_output = tgt_output[:self.tgt_length]

        while len(tgt_input) < self.tgt_length:
            tgt_input.append(self.padding_token)
        while len(tgt_output) < self.tgt_length:
            tgt_output.append(self.padding_token)

        tgt_input = self.tgt_tokenizer.convert_tokens_to_ids(tgt_input)
        tgt_output = self.tgt_tokenizer.convert_tokens_to_ids(tgt_output)

        inp = torch.as_tensor(inp, dtype=torch.long)
        tgt_input = torch.as_tensor(tgt_input, dtype=torch.long)
        tgt_output = torch.as_tensor(tgt_output, dtype=torch.long)

        return inp, tgt_input, tgt_output

    def __len__(self):
        return self.dataframe.shape[0]