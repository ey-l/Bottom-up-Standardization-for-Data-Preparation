import re
import yaml
import nltk
import torch
import string
import unidecode
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
print(f'train shape => {_input1.shape}')
print(f'test shape => {_input0.shape}')
(X_train, X_test, y_train, y_test) = train_test_split(_input1.text.values, _input1.target.values, stratify=_input1.target.values, test_size=0.2, random_state=1)

class Data_gen(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        items = {key: torch.tensor(val[index]) for (key, val) in self.encodings.items()}
        if self.labels:
            items['labels'] = torch.tensor(self.labels[index])
        return items

    def __len__(self):
        return len(self.encodings['input_ids'])
batch_size = 8 * torch.cuda.device_count() if torch.cuda.device_count() > 0 else 8
batch_size
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
train_encoding = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=84)
valid_encoding = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=84)
train_set = Data_gen(train_encoding, y_train.tolist())
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_set = Data_gen(valid_encoding, y_test.tolist())
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
total_steps = len(train_loader) * 3
model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-05)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

class Trainer:

    def __init__(self, model, epochs, scheduler, device, optimizer, auto_break_n):
        self.model = model
        self.epochs = epochs
        self.device = device
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.auto_break_n = auto_break_n
        self.history_loss_train = []
        self.history_loss_valid = []
        self.history_score_train = []
        self.history_score_valid = []
        self.no_improvement_epoch = 0
        self.best_valid_score = 0
        self.model.to(self.device)

    def fit(self, train_loader, valid_loader=None):
        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            preds = []
            targets = []
            for batch in tqdm(train_loader):
                inputs = {k: v.to(self.device) for (k, v) in batch.items()}
                target = batch['labels']
                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                logits = outputs.logits
                loss = outputs.loss
                losses.append(loss.item())
                logits = logits.detach().cpu().numpy()
                targets.extend(target.tolist())
                preds.extend(np.argmax(logits, axis=1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            train_loss = np.mean(losses)
            train_score = f1_score(preds, targets)
            self.history_loss_train.append(train_loss)
            self.history_score_train.append(train_score)
            print('=' * 90)
            print(f'\x1b[1;34m=> {epoch + 1} <= epoch')
            print(f'\x1b[1;31m Train Loss: {train_loss}, Score: {train_score}')
            print('- ' * 45)
            if valid_loader:
                (valid_loss, valid_score) = self.test_score(valid_loader)
                self.history_loss_valid.append(valid_loss)
                self.history_score_valid.append(valid_score)
                print(f'\x1b[1;31m Validation Loss : {valid_loss}, Score: {valid_score}')
                if self.scheduler:
                    self.scheduler.step()
                if self.history_score_valid[-1] > self.best_valid_score:
                    self.no_improvement_epoch = 0
                    self.best_valid_score = self.history_score_valid[-1]
                    self.save_model(epoch + 1)
                else:
                    self.no_improvement_epoch += 1
                print(f'no improvement_iter = {self.no_improvement_epoch}')
                if self.no_improvement_epoch == self.auto_break_n:
                    self.save_model(epoch)
                    print('Auto_break !!!')
                    break

    def test_score(self, valid_loader):
        self.model.eval()
        losses = []
        targets = []
        preds = []
        for batch in tqdm(valid_loader):
            with torch.no_grad():
                inputs = {k: v.to(self.device) for (k, v) in batch.items()}
                target = batch['labels']
                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                logits = outputs.logits
                logits = logits.detach().cpu().numpy()
                loss = outputs.loss.to(self.device)
                losses.append(loss.item())
                targets.extend(target.tolist())
                preds.extend(np.argmax(logits, axis=1))
        return (np.mean(losses), f1_score(preds, targets))

    def save_model(self, n_epoch):
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'best_valid_score': self.best_valid_score, 'n_epoch': n_epoch}, 'model.pth')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device
my_model = Trainer(model, 4, scheduler, device, optimizer, 2)