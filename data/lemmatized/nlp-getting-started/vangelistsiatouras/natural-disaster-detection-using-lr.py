import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import urllib.parse
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input0.head()
punctuation_regex = re.compile('[^\\w\\s]+')
urls_regex = re.compile('(https?:\\/\\/(?:www\\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\\.[^\\s]{2,}|www\\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\\.[^\\s]{2,}|https?:\\/\\/(?:www\\.|(?!www))[a-zA-Z0-9]+\\.[^\\s]{2,}|www\\.[a-zA-Z0-9]+\\.[^\\s]{2,})')

def data_clean(df):
    df['cleaned_text'] = df['text']
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: urls_regex.sub('', str(x)))
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: punctuation_regex.sub('', str(x)))
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub('#', '', str(x)))
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join([item for item in x.split() if item not in ENGLISH_STOP_WORDS]))
    df['cleaned_text'] = df['cleaned_text'].str.lower()
    df['keyword'] = df['keyword'].replace(np.nan, '')
    df['keyword'] = df['keyword'].apply(lambda x: urllib.parse.unquote(str(x)))
    df['keyword'] = df['keyword'].str.lower()
    df['location'] = df['location'].replace(np.nan, '')
    df['location'] = df['location'].apply(lambda x: urllib.parse.unquote(str(x)))
    df['location'] = df['location'].apply(lambda x: punctuation_regex.sub('', str(x)))
    df['location'] = df['location'].str.lower()
    return df

def feature_generate(df):
    df['num_of_words'] = df['text'].str.split().str.len()
    df['num_of_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(str(x))))
    df['num_of_hashtags'] = df['text'].apply(lambda x: len(re.findall('#(\\w+)', str(x))))
    df['num_of_references'] = df['text'].apply(lambda x: len(re.findall('@(\\w+)', str(x))))
    df['num_of_urls'] = df['text'].apply(lambda x: len(re.findall(urls_regex, str(x))))
    return df
fig = plt.figure(figsize=(10, 5))
sns.countplot(x=_input1['target'], data=_input1)
plt.title('Tweet count for each class')
true_df = _input1[_input1['target'] == 1]
fake_df = _input1[_input1['target'] == 0]
c = true_df['location'].value_counts()
print(f"Most common words for 'location' field in true tweets\n{c}")
c = fake_df['location'].value_counts()
print(f"Most common words for 'location' field in fake tweets\n{c}")
c = true_df['keyword'].value_counts()
print(f"Most common words for 'keyword' field in true tweets\n{c}")
c = fake_df['keyword'].value_counts()
print(f"Most common words for 'keyword' field in fake tweets\n{c}")
_input1 = data_clean(_input1)
_input1 = feature_generate(_input1)
_input1
fig = plt.figure(figsize=(10, 5))
sns.histplot(data=_input1, x='num_of_words')
plt.title('Number of Words')
g = sns.FacetGrid(_input1, col='target', height=4)
g.map(plt.hist, 'num_of_words', bins=50)
fig = plt.figure(figsize=(10, 5))
sns.histplot(data=_input1, x='num_of_sentences')
plt.title('Number of Sentences')
g = sns.FacetGrid(_input1, col='target', height=4)
g.map(plt.hist, 'num_of_sentences', bins=20)
fig = plt.figure(figsize=(10, 5))
sns.histplot(data=_input1, x='num_of_hashtags')
plt.title('Number of Hashtags')
g = sns.FacetGrid(_input1, col='target', height=4)
g.map(plt.hist, 'num_of_hashtags', bins=20)
fig = plt.figure(figsize=(10, 5))
sns.histplot(data=_input1, x='num_of_references')
plt.title('Number of References')
g = sns.FacetGrid(_input1, col='target', height=4)
g.map(plt.hist, 'num_of_references', bins=20)
fig = plt.figure(figsize=(10, 5))
sns.histplot(data=_input1, x='num_of_urls')
plt.title('Number of URLs')
g = sns.FacetGrid(_input1, col='target', height=4)
g.map(plt.hist, 'num_of_urls', bins=20)
col_transform = ColumnTransformer(transformers=[('keyword_bow', TfidfVectorizer(strip_accents='unicode', analyzer='word', max_features=400, ngram_range=(1, 1)), 'keyword'), ('location_bow', TfidfVectorizer(strip_accents='unicode', analyzer='word', max_features=1000, ngram_range=(1, 1)), 'location'), ('text_bow1', TfidfVectorizer(analyzer='word', max_features=20000, ngram_range=(1, 1)), 'cleaned_text'), ('text_bow2', TfidfVectorizer(analyzer='word', max_features=50000, ngram_range=(1, 3)), 'cleaned_text'), ('chars_bow2', TfidfVectorizer(analyzer='char', max_features=90000, ngram_range=(1, 6)), 'cleaned_text'), ('scaled', MinMaxScaler(feature_range=(0, 1)), ['num_of_sentences', 'num_of_hashtags', 'num_of_references', 'num_of_urls'])], remainder='drop', verbose_feature_names_out=False)
pipeline = Pipeline([('col_transform', col_transform), ('clf', LogisticRegression(solver='lbfgs', max_iter=1000, n_jobs=-1))])
X = _input1[['keyword', 'location', 'cleaned_text', 'num_of_sentences', 'num_of_hashtags', 'num_of_references', 'num_of_urls']]
y = _input1['target']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=10)
X_train
folds = KFold(n_splits=5, shuffle=True, random_state=10)
scores = cross_validate(pipeline, X_train, y_train, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], cv=5, n_jobs=-1, return_train_score=False)
print(f'Cross validation scores \n{scores}')