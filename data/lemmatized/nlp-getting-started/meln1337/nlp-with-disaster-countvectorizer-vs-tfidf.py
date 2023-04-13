import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1.info()
_input1.shape
(_input1['keyword'].nunique(), _input1['location'].nunique())
for com in _input1.query('target == 1')['text'].head(5):
    print(com)
for com in _input1.query('target == 0')['text'].head(5):
    print(com)
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input0.head()
_input0.shape
test_id = _input0['id']
df = _input1.append(_input0).reset_index()
df.shape
df['keyword'] = df['keyword'].fillna('Unknown', inplace=False)
df['location'] = df['location'].fillna('Unknown', inplace=False)
df
df.isna().sum()
temp_df = _input1.copy()
temp_df['text_length'] = _input1['text'].apply(lambda x: len(x))
sns.histplot(data=temp_df, x='text_length', hue='target', kde=True)
plt.title('Distribution of text length')
plt.title('Count plot of target')
sns.countplot(data=_input1, x='target')
plt.figure(figsize=(21, 150))
sns.countplot(data=_input1, y='keyword', hue='target')
plt.title('Keyword number')
_input1['keyword'] = _input1['keyword'].astype('category')
_input1['location'] = _input1['location'].astype('category')
_input1['location'] = _input1['location'].apply(lambda x: x.replace('$$', ''))
import re
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
import spacy
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
len(STOPWORDS)

def preprocess_text(text, stem=False):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9 ]', '', text)
    text = re.sub('http\\S+', '', text)
    text = re.sub('\\s\\s+', ' ', text)
    text = re.sub('[0-9]+', '', text)
    text = text.split(' ')
    text = ' '.join([word for word in text if not word in STOPWORDS])
    text = remove_stopwords(text)
    if stem:
        stem = SnowballStemmer(language='english')
        text = ' '.join([stem.stem(word) for word in text.split(' ')])
    else:
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split(' ')])
    text = text.strip()
    return text

def tokenize_word(text, stem=True):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9 ]', '', text)
    text = re.sub('http\\S+', '', text)
    text = re.sub('\\s\\s+', ' ', text)
    text = re.sub('[0-9]+', '', text)
    text = text.split(' ')
    text = ' '.join([word for word in text if not word in STOPWORDS])
    text = remove_stopwords(text)
    if stem:
        stem = SnowballStemmer(language='english')
        text = ' '.join([stem.stem(word) for word in text.split(' ')])
    else:
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split(' ')])
    text = text.strip()
    return text.split(' ')
preprocess_text(df['text'][0], stem=True)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import nltk
nltk.download('omw-1.4')
cv_vectorizer = CountVectorizer()