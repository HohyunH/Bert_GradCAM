import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')

class BERT_CNN(Dataset):
    def __init__(self, df, tokenizer):
        def data_setting(df, max_len, tokenizer):
            # [CLS] [SEP] 붙임, token 변환
            description = list(df["review"])
            descriptions = [get_token(x, tokenizer) for x in description]
            description = [''.join(d) for d in descriptions]
            # max_len
            max_len = max_len

            # convert Id + padding
            input_ids = [get_ids(x, max_len, tokenizer) for x in descriptions]

            # Attention_masks
            attention_masks = []
            attention_masks = get_mask(input_ids)

            # labels
            target = list(df['label'])

            # class수
            num_classes = len(df['label'].unique())

            return input_ids, target, attention_masks, description

        self.tokenizer = tokenizer

        input_ids, input_labels, attention_masks, description = data_setting(df, 512, self.tokenizer)

        self.reviews = description
        self.inputs = input_ids
        self.labels = input_labels
        self.masks = attention_masks

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        text = torch.tensor(self.inputs[idx])
        label = self.labels[idx]
        review = self.reviews[idx]
        mask = torch.LongTensor(self.masks[idx])

        return text, label, mask, review

    # 1. 모든 sentence에 CLS, SEP 붙인다.
def get_token(text, tokenizer):
    text = "[CLS] " + " [SEP] ".join(sent_tokenize(text)) + " [SEP]"
    tokenized_text = tokenizer.tokenize(text)
    return tokenized_text

# 2. token index로 변환, padding 한다.
def get_ids(tokenized_text, max_length, tokenizer):
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    if len(indexed_tokens) < max_length:
        input_ids = indexed_tokens + [0] * (max_length - len(indexed_tokens))
    else:
        input_ids = indexed_tokens[:max_length]

    return input_ids

# 3. segment_ids를 만든다.
def get_mask(input_ids):
    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks

def class_same(train_df, class_num):
    zero_df = train_df[train_df['label'] == 0][:class_num]
    one_df = train_df[train_df['label'] == 1][:class_num]
    total_df = pd.concat([zero_df, one_df])
    total_df = total_df.sample(frac=1).reset_index(drop=True)

    return total_df

if __name__ == '__main__':
    torch.manual_seed(777)
    random.seed(777)
    np.random.seed(777)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = list(tokenizer.vocab.keys())
    word_dict = dict(tokenizer.vocab.items())

    yelp = pd.read_csv('./yelp.csv')
    yelp['label'] = yelp['rating'] - 1
    del yelp['rating']

    class_num = 100

    train_df = yelp[:1000]
    test_df = yelp[1000:-1000]
    val_df = yelp[-1000:]

    train_df = class_same(train_df, class_num)
    test_df_1 = test_df[:20000]

    title_order = ["NEG", "POS"]

    train_data = BERT_CNN(train_df, tokenizer)
    train_loader = DataLoader(train_data, batch_size=2, num_workers=2, shuffle=True)

    test_data = BERT_CNN(test_df_1, tokenizer)
    test_loader = DataLoader(test_data, batch_size=2, num_workers=2, shuffle=False)

    val_data = BERT_CNN(val_df, tokenizer)
    val_loader = DataLoader(val_data, batch_size=2, num_workers=2, shuffle=False)

    for batch in train_loader:
        print(batch)
        import sys;

        sys.exit(0)