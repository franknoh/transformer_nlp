import tokenization
import pandas
import random
import json
import torch
import copy

import numpy as np


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
        tgt = self.dataframe.loc[idx, self.tgt_column]
        tokenized_tgt = self.tgt_tokenizer.tokenize(tgt)

        tokenized_src_a = tokenized_src[:self.src_length]
        while len(tokenized_src_a) < self.src_length:
            tokenized_src_a.append(self.padding_token)
        inp_a = self.src_tokenizer.convert_tokens_to_ids(tokenized_src_a)
        tgt_token_a = [self.start_token]
        tgt_token_a.extend(tokenized_tgt)
        tgt_token_a.append(self.end_token)
        tgt_input_a = tgt_token_a[:-1]
        tgt_output_a = tgt_token_a[1:]
        tgt_input_a = tgt_input_a[:self.tgt_length]
        tgt_output_a = tgt_output_a[:self.tgt_length]
        while len(tgt_input_a) < self.tgt_length + 1:
            tgt_input_a.append(self.padding_token)
        while len(tgt_output_a) < self.tgt_length + 1:
            tgt_output_a.append(self.padding_token)
        tgt_input_a = self.tgt_tokenizer.convert_tokens_to_ids(tgt_input_a)
        tgt_output_a = self.tgt_tokenizer.convert_tokens_to_ids(tgt_output_a)
        inp_a = torch.as_tensor(inp_a, dtype=torch.long)
        tgt_input_a = torch.as_tensor(tgt_input_a, dtype=torch.long)
        tgt_output_a = torch.as_tensor(tgt_output_a, dtype=torch.long)

        tokenized_src_b = tokenized_tgt[self.tgt_length:]
        while len(tokenized_src_b) < self.tgt_length:
            tokenized_src_b.append(self.padding_token)
        inp_b = self.tgt_tokenizer.convert_tokens_to_ids(tokenized_src_b)
        tgt_token_b = [self.start_token]
        tgt_token_b.extend(tokenized_src)
        tgt_token_b.append(self.end_token)
        tgt_input_b = tgt_token_b[:-1]
        tgt_output_b = tgt_token_b[1:]
        tgt_input_b = tgt_input_b[:self.src_length]
        tgt_output_b = tgt_output_b[:self.src_length]
        while len(tgt_input_b) < self.src_length + 1:
            tgt_input_b.append(self.padding_token)
        while len(tgt_output_b) < self.src_length + 1:
            tgt_output_b.append(self.padding_token)
        tgt_input_b = self.src_tokenizer.convert_tokens_to_ids(tgt_input_b)
        tgt_output_b = self.src_tokenizer.convert_tokens_to_ids(tgt_output_b)
        inp_b = torch.as_tensor(inp_b, dtype=torch.long)
        tgt_input_b = torch.as_tensor(tgt_input_b, dtype=torch.long)
        tgt_output_b = torch.as_tensor(tgt_output_b, dtype=torch.long)

        return inp_a, tgt_input_a, tgt_output_a, inp_b, tgt_input_b, tgt_output_b

    def __len__(self):
        return self.dataframe.shape[0]
