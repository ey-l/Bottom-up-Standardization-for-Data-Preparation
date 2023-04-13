import pandas as pd
import spacy
from spacy import displacy
import seaborn as sns
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.info()
sns.countplot(y='target', data=_input1, palette='Set3')
sns.countplot(y='location', data=_input1, palette='Set3', order=_input1['location'].value_counts().iloc[:6].index)
_input1['location'] = _input1['location'].replace(['United States', 'New York', 'Los Angeles', 'Los Angeles, CA', 'Washington, DC'], 'USA')
_input1['location'] = _input1['location'].replace(['London'], 'UK')
_input1['location'] = _input1['location'].replace(['Mumbai'], 'India')
sns.countplot(y='location', data=_input1, palette='Set3', order=_input1['location'].value_counts().iloc[:6].index)
sns.countplot(y='keyword', data=_input1, palette='Set2', order=_input1['keyword'].value_counts().iloc[:3].index)
nlp = spacy.load('en_core_web_sm')
doc = nlp(_input1['text'][58])
displacy.render(doc, style='ent')
doc = nlp(_input1['text'][10])
displacy.render(doc, style='ent')
tokenized_text = pd.DataFrame()
for (i, token) in enumerate(doc):
    tokenized_text.loc[i, 'text'] = token.text
    tokenized_text.loc[i, 'type'] = token.pos_
    tokenized_text.loc[i, 'lemma'] = (token.lemma_,)
    tokenized_text.loc[i, 'is_alphabetic'] = token.is_alpha
    tokenized_text.loc[i, 'is_stop'] = token.is_stop
    tokenized_text.loc[i, 'is_punctuation'] = token.is_punct
    tokenized_text.loc[i, 'sentiment'] = token.sentiment
tokenized_text[:30]
displacy.render(doc, style='dep', jupyter='true')
spacy.explain('ADP')
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS
tokenizer = English()

def text_tokenizer(sentence):
    tokens = tokenizer(sentence)
    tokens = [word.lemma_.lower().strip() if word.lemma_ != '-PRON-' else word.lower_ for word in tokens]
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations]
    return tokens
from sklearn.base import TransformerMixin

class CleanTextTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

def clean_text(text):
    text = text.strip().replace('\n', ' ').replace('\r', ' ')
    text = text.lower()
    return text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer = CountVectorizer(tokenizer=text_tokenizer, ngram_range=(1, 1))
X = _input1['text']
y = _input1['target']
from sklearn.model_selection import train_test_split
(X_train, X_vald, y_train, y_vald) = train_test_split(X, y, test_size=0.15)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
classifier = LinearSVC()
pipeline = Pipeline([('cleaner', CleanTextTransformer()), ('vectorizer', vectorizer), ('classifier', classifier)])