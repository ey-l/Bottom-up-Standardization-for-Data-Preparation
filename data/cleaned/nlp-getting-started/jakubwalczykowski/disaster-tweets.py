

import os
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import pkg_resources
from symspellpy import SymSpell, Verbosity
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import pickle
from emot.emo_unicode import UNICODE_EMOJI
from emot.emo_unicode import EMOTICONS_EMO
from wordcloud import WordCloud
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
sub_df = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
train_df
train_df['target'].hist()
train_df.head(10)
train_df.info()
test_df.info()

def visualize(label):
    words = ''
    for msg in train_df[train_df['target'] == label]['text']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')

visualize(0)
visualize(1)
train_df[train_df['target'] == 0]['text'].values[0]
train_df[train_df['target'] == 1]['text'].values[0]
sub_df.head()

def convert_emojis(text):
    for emot in UNICODE_EMOJI:
        text = text.replace(emot, '_'.join(UNICODE_EMOJI[emot].replace(',', '').replace(':', '').split()))
    return text
train_df['text'] = train_df['text'].apply(convert_emojis)
test_df['text'] = test_df['text'].apply(convert_emojis)
train_df['keyword'] = train_df['keyword'].fillna('')
train_df['text'] = train_df['keyword'] + ' ' + train_df['text']
test_df['keyword'] = test_df['keyword'].fillna('')
test_df['text'] = test_df['keyword'] + ' ' + test_df['text']
train_df = train_df.drop(['keyword', 'location'], axis=1)
test_df = test_df.drop(['keyword', 'location'], axis=1)
train_df

def remove_html(text):
    soup = BeautifulSoup(text)
    text = soup.get_text()
    return text
train_df['text'] = train_df['text'].apply(remove_html)
test_df['text'] = test_df['text'].apply(remove_html)
train_df
train_df['text'] = train_df['text'].str.lower()
test_df['text'] = test_df['text'].str.lower()
train_df

def remove_urls(text):
    pattern = re.compile('https?://(www\\.)?(\\w+)(\\.\\w+)(/\\w*)?')
    text = re.sub(pattern, '', text)
    return text
train_df['text'] = train_df['text'].apply(remove_urls)
test_df['text'] = test_df['text'].apply(remove_urls)
train_df

def remove_mentions(text):
    pattern = re.compile('@\\w+')
    text = re.sub(pattern, '', text)
    return text
train_df['text'] = train_df['text'].apply(remove_mentions)
test_df['text'] = test_df['text'].apply(remove_mentions)
train_df

def remove_unicode_chars(text):
    text = text.encode('ascii', 'ignore').decode()
    return text
train_df['text'] = train_df['text'].apply(remove_unicode_chars)
test_df['text'] = test_df['text'].apply(remove_unicode_chars)
train_df
string.punctuation

def remove_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    return text
train_df['text'] = train_df['text'].apply(remove_punctuations)
test_df['text'] = test_df['text'].apply(remove_punctuations)
train_df

def remove_extra_spaces(text):
    text = re.sub(' +', ' ', text).strip()
    return text
train_df['text'] = train_df['text'].apply(remove_extra_spaces)
test_df['text'] = test_df['text'].apply(remove_extra_spaces)
train_df
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename('symspellpy', 'frequency_dictionary_en_82_765.txt')
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def correct_spelling_symspell(text):
    words = [sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)[0].term for word in text.split()]
    text = ' '.join(words)
    return text
train_df['text'] = train_df['text'].apply(correct_spelling_symspell)
test_df['text'] = test_df['text'].apply(correct_spelling_symspell)
train_df
bigram_path = pkg_resources.resource_filename('symspellpy', 'frequency_bigramdictionary_en_243_342.txt')
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

def correct_spelling_symspell_compound(text):
    words = [sym_spell.lookup_compound(word, max_edit_distance=2)[0].term for word in text.split()]
    text = ' '.join(words)
    return text
train_df['text'] = train_df['text'].apply(correct_spelling_symspell_compound)
test_df['text'] = test_df['text'].apply(correct_spelling_symspell_compound)
train_df
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join([word for word in str(text).split() if word not in stop_words])
train_df['text'] = train_df['text'].apply(remove_stopwords)
test_df['text'] = test_df['text'].apply(remove_stopwords)
train_df
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = [lemmatizer.lemmatize(word) for word in text.split()]
    text = ' '.join(words)
    return text
train_df['text'] = train_df['text'].apply(lemmatize_text)
test_df['text'] = test_df['text'].apply(lemmatize_text)
train_df
visualize(0)
visualize(1)
tfidf = TfidfVectorizer(max_features=105325, binary=True, analyzer='word', ngram_range=(1, 3), use_idf=True, smooth_idf=1, sublinear_tf=1)
X = tfidf.fit_transform(train_df['text']).toarray()
y = train_df['target']
(tfidfX_train, tfidfX_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)
print(tfidfX_train.shape)
print(tfidfX_test.shape)
tfidfX_test
clf1 = LogisticRegression()