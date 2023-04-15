import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
bert_train = train
train.sample(10)
train.describe()
for col in train.columns:
    print(col + ' column has: ' + str(round(train[col].isna().sum() / train[col].isna().count() * 100, 2)) + '% Missing values')
train.drop(['keyword', 'location'], inplace=True, axis=1)
train.head(1)
train['target'].value_counts().plot(kind='bar')
from bs4 import BeautifulSoup
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm

def text_cleaning(text):
    text = text.lower()
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub('https?://[A-Za-z0-9./]+', '', text)
    text = re.sub('@[A-Za-z0-9]+', '', text)
    text = re.sub('\\W+|_', ' ', text)
    text = word_tokenize(text)
    lm = WordNetLemmatizer()
    words = [lm.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    return ' '.join(words)
train.text = train['text'].progress_apply(text_cleaning)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 1), max_df=1.0, min_df=1)
X = tfidf.fit_transform(train['text'])
print(X.shape)
y = train.target
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression()