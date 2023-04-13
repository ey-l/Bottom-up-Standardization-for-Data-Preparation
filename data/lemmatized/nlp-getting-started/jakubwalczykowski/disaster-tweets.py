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
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input1
_input1['target'].hist()
_input1.head(10)
_input1.info()
_input0.info()

def visualize(label):
    words = ''
    for msg in _input1[_input1['target'] == label]['text']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
visualize(0)
visualize(1)
_input1[_input1['target'] == 0]['text'].values[0]
_input1[_input1['target'] == 1]['text'].values[0]
_input2.head()

def convert_emojis(text):
    for emot in UNICODE_EMOJI:
        text = text.replace(emot, '_'.join(UNICODE_EMOJI[emot].replace(',', '').replace(':', '').split()))
    return text
_input1['text'] = _input1['text'].apply(convert_emojis)
_input0['text'] = _input0['text'].apply(convert_emojis)
_input1['keyword'] = _input1['keyword'].fillna('')
_input1['text'] = _input1['keyword'] + ' ' + _input1['text']
_input0['keyword'] = _input0['keyword'].fillna('')
_input0['text'] = _input0['keyword'] + ' ' + _input0['text']
_input1 = _input1.drop(['keyword', 'location'], axis=1)
_input0 = _input0.drop(['keyword', 'location'], axis=1)
_input1

def remove_html(text):
    soup = BeautifulSoup(text)
    text = soup.get_text()
    return text
_input1['text'] = _input1['text'].apply(remove_html)
_input0['text'] = _input0['text'].apply(remove_html)
_input1
_input1['text'] = _input1['text'].str.lower()
_input0['text'] = _input0['text'].str.lower()
_input1

def remove_urls(text):
    pattern = re.compile('https?://(www\\.)?(\\w+)(\\.\\w+)(/\\w*)?')
    text = re.sub(pattern, '', text)
    return text
_input1['text'] = _input1['text'].apply(remove_urls)
_input0['text'] = _input0['text'].apply(remove_urls)
_input1

def remove_mentions(text):
    pattern = re.compile('@\\w+')
    text = re.sub(pattern, '', text)
    return text
_input1['text'] = _input1['text'].apply(remove_mentions)
_input0['text'] = _input0['text'].apply(remove_mentions)
_input1

def remove_unicode_chars(text):
    text = text.encode('ascii', 'ignore').decode()
    return text
_input1['text'] = _input1['text'].apply(remove_unicode_chars)
_input0['text'] = _input0['text'].apply(remove_unicode_chars)
_input1
string.punctuation

def remove_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    return text
_input1['text'] = _input1['text'].apply(remove_punctuations)
_input0['text'] = _input0['text'].apply(remove_punctuations)
_input1

def remove_extra_spaces(text):
    text = re.sub(' +', ' ', text).strip()
    return text
_input1['text'] = _input1['text'].apply(remove_extra_spaces)
_input0['text'] = _input0['text'].apply(remove_extra_spaces)
_input1
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename('symspellpy', 'frequency_dictionary_en_82_765.txt')
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def correct_spelling_symspell(text):
    words = [sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)[0].term for word in text.split()]
    text = ' '.join(words)
    return text
_input1['text'] = _input1['text'].apply(correct_spelling_symspell)
_input0['text'] = _input0['text'].apply(correct_spelling_symspell)
_input1
bigram_path = pkg_resources.resource_filename('symspellpy', 'frequency_bigramdictionary_en_243_342.txt')
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

def correct_spelling_symspell_compound(text):
    words = [sym_spell.lookup_compound(word, max_edit_distance=2)[0].term for word in text.split()]
    text = ' '.join(words)
    return text
_input1['text'] = _input1['text'].apply(correct_spelling_symspell_compound)
_input0['text'] = _input0['text'].apply(correct_spelling_symspell_compound)
_input1
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join([word for word in str(text).split() if word not in stop_words])
_input1['text'] = _input1['text'].apply(remove_stopwords)
_input0['text'] = _input0['text'].apply(remove_stopwords)
_input1
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = [lemmatizer.lemmatize(word) for word in text.split()]
    text = ' '.join(words)
    return text
_input1['text'] = _input1['text'].apply(lemmatize_text)
_input0['text'] = _input0['text'].apply(lemmatize_text)
_input1
visualize(0)
visualize(1)
tfidf = TfidfVectorizer(max_features=105325, binary=True, analyzer='word', ngram_range=(1, 3), use_idf=True, smooth_idf=1, sublinear_tf=1)
X = tfidf.fit_transform(_input1['text']).toarray()
y = _input1['target']
(tfidfX_train, tfidfX_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)
print(tfidfX_train.shape)
print(tfidfX_test.shape)
tfidfX_test
clf1 = LogisticRegression()