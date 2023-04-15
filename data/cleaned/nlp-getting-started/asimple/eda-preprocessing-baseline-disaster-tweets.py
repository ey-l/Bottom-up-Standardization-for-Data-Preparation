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

    class model:
        params = dict(C=1, l1_ratio=0.2, solver='saga', penalty='elasticnet', random_state=42)
train_data = pd.read_csv(CFG.path.to_train_data)
test_data = pd.read_csv(CFG.path.to_test_data)
print(f'train shape => {train_data.shape}')
print(f'test shape => {test_data.shape}')
train_data.sample(5)
test_data.sample(5)
train_data.describe(include=['O'])
test_data.describe(include=['O'])

def missing_values(data: pd.DataFrame) -> pd.DataFrame:
    miss_value_percent = data.isna().sum() / data.shape[0] * 100
    return pd.DataFrame(miss_value_percent, columns=['Missing_percent']).query('Missing_percent > 0')
missing_values(train_data)
missing_values(test_data)
sns.countplot(train_data.target)
plt.ylabel('Samples')
sns.barplot(y=train_data['location'].value_counts()[:10].index, x=train_data['location'].value_counts()[:10])
{target: train_data.query(f'target == {target}').text.values[0] for target in train_data.target.unique()}
(X_train, X_test, y_train, y_test) = train_test_split(train_data.text.values, train_data.target.values, stratify=train_data.target.values, test_size=0.2, random_state=1)

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
prep_features = Pipeline(steps=[('Preprocessing', FunctionTransformer(piepline_preprocess)), ('tfidf', TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2)))])
model = LogisticRegression(**CFG.model.params)
clf = Pipeline(steps=[('preprocessor', prep_features), ('model', model)])