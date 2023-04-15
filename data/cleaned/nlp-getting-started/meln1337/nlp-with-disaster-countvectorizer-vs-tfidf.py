import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
train_df.head()
train_df.info()
train_df.shape
(train_df['keyword'].nunique(), train_df['location'].nunique())
for com in train_df.query('target == 1')['text'].head(5):
    print(com)
for com in train_df.query('target == 0')['text'].head(5):
    print(com)
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
test_df.head()
test_df.shape
test_id = test_df['id']
df = train_df.append(test_df).reset_index()
df.shape
df['keyword'].fillna('Unknown', inplace=True)
df['location'].fillna('Unknown', inplace=True)
df
df.isna().sum()
temp_df = train_df.copy()
temp_df['text_length'] = train_df['text'].apply(lambda x: len(x))
sns.histplot(data=temp_df, x='text_length', hue='target', kde=True)
plt.title('Distribution of text length')
plt.title('Count plot of target')
sns.countplot(data=train_df, x='target')
plt.figure(figsize=(21, 150))
sns.countplot(data=train_df, y='keyword', hue='target')
plt.title('Keyword number')
train_df['keyword'] = train_df['keyword'].astype('category')
train_df['location'] = train_df['location'].astype('category')
train_df['location'] = train_df['location'].apply(lambda x: x.replace('$$', ''))
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