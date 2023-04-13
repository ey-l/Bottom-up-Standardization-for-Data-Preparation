import numpy as np
import pandas as pd
import os
import random
import glob
import re
from torch import no_grad
from torch import tensor
from torch.nn import Softmax
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.cuda import is_available
import pytorch_lightning as pl
MODEL_NAME = 'vinai/bertweet-base'
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import gc
import pickle
import sys
import psutil
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv', index_col='id')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv', index_col='id')
for df in [_input1, _input0]:
    df['text'] = df['text'].apply(lambda x: re.sub('https?://[A-Za-z0-9/:.]+', '(url)', x))
    df['text'] = df['text'].apply(lambda x: re.sub('@[A-Za-z0-9/:._]+', '@user', x))
    df['text'] = df['text'].apply(lambda x: re.sub('[0-9]+', '0', x))
    df['text'] = df['text'].apply(lambda x: re.sub('\\s+', ' ', x))
    df['keyword'] = df['keyword'].str.replace('%20', '_')
    df['text'] = df['text'] + df['keyword'].apply(lambda x: ' #' + x if not x is np.nan else '')
_input1[_input1['text'].str.contains('http')]
_input1[_input1['text'].str.contains('@')]
all_data = pd.concat([_input1, _input0])
all_data[all_data.duplicated('text')].sort_values('text').head(50)
print(_input1['keyword'].nunique(), _input0['keyword'].nunique())
plt.pie(_input1['target'].value_counts(), labels=['no', 'yes'], startangle=90, autopct='%.2f%%')
(fig, ax) = plt.subplots(1, 2, figsize=(20, 40))
sns.histplot(y=_input1['keyword'].sort_values(), hue=_input1['target'], multiple='stack', ax=ax[0])
sns.histplot(y=_input0['keyword'].sort_values(), ax=ax[1])
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
print(len(tokenizer))
tokenizer.add_tokens(['(url)', '@user'], special_tokens=True)
print(len(tokenizer))
max_length = 80
dataset_for_loader = []
for idx in _input1.index:
    encoding = tokenizer(_input1.loc[idx, 'text'], max_length=max_length, return_token_type_ids=False, padding='max_length', truncation=True)
    encoding['labels'] = _input1.loc[idx, 'target']
    encoding = {k: tensor(v) for (k, v) in encoding.items()}
    dataset_for_loader.append(encoding)
dataset_for_eval = []
for idx in _input0.index:
    encoding = tokenizer(_input0.loc[idx, 'text'], max_length=max_length, return_token_type_ids=False, padding='max_length', truncation=True)
    encoding = {k: tensor(v) for (k, v) in encoding.items()}
    dataset_for_eval.append(encoding)
ids_len = []
for data in dataset_for_loader:
    ids_len.append(sum(data['attention_mask']))
print(max(ids_len))
wordlist = []
for dataset in dataset_for_loader:
    wordlist += dataset['input_ids'].tolist()
from collections import Counter
df_wordfreq = pd.DataFrame(Counter(wordlist).items(), columns=['word_id', 'freq'])
df_wordfreq['word'] = df_wordfreq['word_id'].apply(lambda x: tokenizer.decode(x))
df_wordfreq.sort_values('freq', ascending=False).head(50)

class BertForSequenceClassification_pl(pl.LightningModule):

    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_sc = AutoModel.from_pretrained(model_name)
        self.bert_sc.resize_token_embeddings(len(tokenizer))
        self.hidden_size = [*self.bert_sc.parameters()][-1].size()[0]
        self.cnn1 = torch.nn.Conv1d(max_length, 16, kernel_size=3, padding=1)
        self.maxpool = torch.nn.MaxPool1d(16)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.regressor = torch.nn.Linear(self.hidden_size, 2)
        self.criterion = torch.nn.CrossEntropyLoss(weight=tensor([1, 0.5703 / 0.4297]))

    def forward(self, batch):
        outputs = self.bert_sc(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).last_hidden_state
        outputs = self.cnn1(outputs).max(dim=1).values.squeeze(1)
        outputs = self.dropout(outputs)
        outputs = self.regressor(outputs)
        return outputs

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.criterion(output, batch['labels'])
        with no_grad():
            return loss

    def validation_step(self, batch, batch_idx):
        with no_grad():
            output = self(batch)
            loss = self.criterion(output, batch['labels'])
            self.log('val_loss', loss)

    def predict_step(self, batch, batch_idx):
        with no_grad():
            output = self(batch)
            s = Softmax()
            output = s(output)
            return output

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3, shuffle=True)
random.seed(0)
random.shuffle(dataset_for_loader)
random.seed(0)
dataset_idx = _input1.index.values
random.shuffle(dataset_idx)
n = len(dataset_for_loader)
accelerator = 'gpu' if is_available() else 'cpu'
import torch
from sklearn.metrics import f1_score, confusion_matrix
thresholds = []
for (i, (train_index, test_index)) in enumerate(skf.split(_input1, _input1['target'], _input1['target'])):
    print(f"train_data:{_input1.iloc[train_index]['target'].sum()},test_data:{_input1.iloc[test_index]['target'].sum()}")
    dataset_train = torch.utils.data.Subset(dataset_for_loader, train_index)
    dataset_val = torch.utils.data.Subset(dataset_for_loader, test_index)
    dataloader_train = DataLoader(dataset_train, batch_size=32, num_workers=2, pin_memory=True, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=64, num_workers=2, pin_memory=True)
    earlystopping = pl.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=True, mode='min', strict=True)
    checkpoint = pl.callbacks.ModelCheckpoint(save_top_k=1, monitor='val_loss')
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=30, callbacks=[earlystopping, checkpoint], default_root_dir=f'../working/model_{i}')
    model = BertForSequenceClassification_pl(model_name=MODEL_NAME, num_labels=2, lr=2e-06)