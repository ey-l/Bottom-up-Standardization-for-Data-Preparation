import re
import yaml
import nltk
import string
import unidecode
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score

class CFG:

    class path:
        to_test_data = 'data/input/nlp-getting-started/test.csv'
        to_train_data = 'data/input/nlp-getting-started/train.csv'
        to_save_predicts = 'data/input/nlp-getting-started/submission.csv'

    class split:
        seed = 1
        test_size = 0.2
        shuffle = True
        random_state = 1

    class train:
        LogisticRegression = dict(C=1, l1_ratio=0.2, solver='saga', random_state=42, penalty='elasticnet')
        TfidfVectorizer = dict(min_df=1, max_df=0.25, ngram_range=(1, 2))
train_data = pd.read_csv(CFG.path.to_train_data)
test_data = pd.read_csv(CFG.path.to_test_data)
print(f'train shape => {train_data.shape}')
print(f'test shape => {test_data.shape}')
(X_train, X_test, y_train, y_test) = train_test_split(train_data.text.values, train_data.target.values, stratify=train_data.target.values, test_size=CFG.split.test_size, random_state=CFG.split.random_state)

def preprocessing(text: str) -> str:
    text = text.lower()
    text = unidecode.unidecode(text)
    text = re.sub('\\[.*?\\]', '', text)
    text_without_url = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text_without_tag = re.sub('<.*?>+', '', text_without_url)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text_without_tag)
    text = re.sub('\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text

def remove_stopwords(text: List[str]) -> List[str]:
    return [w for w in text if w not in stopwords.words('english')]

def array_to_str(text: List[str]) -> str:
    return ' '.join(text)

def piepline_preprocess(text: List[str]) -> List[str]:
    tokenizer = nltk.tokenize.RegexpTokenizer('\\w+')
    process = [preprocessing, tokenizer.tokenize, remove_stopwords, array_to_str]
    for proc in process:
        text = list(map(proc, text))
    return text
print(f'Before => {train_data.text[0]}')
print(f'After  => {piepline_preprocess([train_data.text[0]])}')
prep_features = Pipeline(steps=[('Preprocessing', FunctionTransformer(piepline_preprocess)), ('tfidf', TfidfVectorizer(**CFG.train.TfidfVectorizer))])
model = LogisticRegression(**CFG.train.LogisticRegression)
clf = Pipeline(steps=[('preprocessor', prep_features), ('model', model)])