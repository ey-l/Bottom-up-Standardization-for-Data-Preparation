import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input1.head()
_input1.isna().sum()
X = _input1['text']
y = _input1.target
print('train text shape', X.shape)
print('train target shape', y.shape)

def display_util(method, sample):
    new_sample = sample.apply(method)
    frame = pd.DataFrame(data={'before': sample, 'After': new_sample})
    return frame
sample = X.sample(n=10)
sample
example = '<p> This is an example <br> hello world <p/>'
re.sub(pattern='<\\w+/?>', repl='', string=example)

def remove_html(text):
    return re.sub(pattern='<\\w+/?>', repl='', string=text)
display_util(remove_html, sample).head(10)
example = '@BritishBakeOff This has opened up a new level of reality show'
re.sub(pattern='@[^\\s]*', repl='', string=example)

def remove_mentioning(text):
    return re.sub(pattern='@[^\\s]*', repl='', string=text)
display_util(remove_mentioning, sample).head(10)
example = 'Free Ebay Sniping RT? http://t.co/B231Ul1O1K  get yours now'
re.sub(pattern='http[^\\s]*', repl='', string=example)

def remove_url(text):
    return re.sub(pattern='http[^\\s]*', repl='', string=text)
display_util(remove_url, sample).head()
print(string.punctuation)
re.escape(string.punctuation)
example = '@president # obama Healthcare plan is a prioroty #obamacare #us-election '
re.sub(pattern='[{0}]'.format(string.punctuation), repl='', string=example)

def remove_punctuation(text):
    return re.sub(pattern='[{0}]'.format(string.punctuation), repl='', string=text)
display_util(remove_punctuation, sample).head(10)
example = '13000 people receive wildfires evacuation order'
re.sub(pattern='\\d+', repl='', string=example)

def remove_numbers(text):
    return re.sub(pattern='\\d+', repl='', string=text)
display_util(remove_numbers, sample)
from nltk.corpus import stopwords
english_stop_words = stopwords.words('english')
print('English stop word', english_stop_words[:5])
example = 'US inflation eases in July as petrol prices drop'
' '.join([word for word in example.split() if word not in english_stop_words])
from nltk.tokenize import word_tokenize

def remove_stop_words(text):
    return ' '.join([word for word in word_tokenize(text) if word not in english_stop_words])
display_util(remove_stop_words, sample).head(10)
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
example = "the boy's cars are different colors"
print([stemmer.stem(ex) for ex in example.split()])

def stemming(text):
    return ' '.join([stemmer.stem(word) for word in word_tokenize(text)])
display_util(stemming, sample).head(10)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
example = "the boy's cars are different colors"
print([lemmatizer.lemmatize(ex, pos='v') for ex in example.split()])

def lemmatize(text, pos='v'):
    return ' '.join([lemmatizer.lemmatize(word, pos=pos) for word in word_tokenize(text)])
display_util(lemmatize, sample).head()

def process_data(X):
    return X.apply(lambda text: text.lower()).apply(remove_html).apply(remove_url).apply(remove_mentioning).apply(remove_punctuation).apply(remove_numbers).apply(remove_stop_words)
X_new = process_data(X)
lemmatized = X_new.apply(lemmatize)
stem = X_new.apply(stemming)
frame = pd.DataFrame(data={'Raw': X, 'Processed': X_new, 'Lemmatized': lemmatized, 'Stemming': stem})
frame.head(10)
count_vectorizer = CountVectorizer(strip_accents='unicode', stop_words='english', ngram_range=(1, 2), max_df=0.9)
tfidf_vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', max_df=0.9)

def build_pipeline(vectorizer=CountVectorizer(), score_fn=chi2, topK=10000, model=MultinomialNB(), feature='Processed'):
    feature = feature
    feature_transformations = Pipeline(steps=[('vectorizer', vectorizer), ('select', SelectKBest(score_func=score_fn, k=topK))])
    transformer = ColumnTransformer(transformers=[('text', feature_transformations, feature)])
    pipe = Pipeline(steps=[('preprocessing', transformer), ('modeling', model)])
    return pipe
model_list = [(MultinomialNB(alpha=0.01), 'Multinomial Naive Bayes'), (BernoulliNB(alpha=0.01), 'Bernoulli Naive Bayes'), (ComplementNB(alpha=0.1), 'Complement Naive Bayes'), (PassiveAggressiveClassifier(max_iter=100, early_stopping=True), 'Passive-Aggressive'), (LinearSVC(penalty='l1', dual=False, tol=0.001), 'LinearSVC with penality = l1'), (LinearSVC(penalty='l2', dual=False, tol=0.001), 'LinearSVC with penality = l2'), (SGDClassifier(alpha=0.0001, max_iter=100, penalty='l1', early_stopping=True), 'SGD Classifier with penality = l1'), (SGDClassifier(alpha=0.0001, max_iter=100, penalty='l2', early_stopping=True), 'SGD Classifier with penality = l2'), (SGDClassifier(alpha=0.0001, max_iter=100, penalty='elasticnet', early_stopping=True), 'SGD Classifier with penality = elasticnet')]

def score(X, y, **kwargs):
    for (model, name) in model_list:
        pipe = build_pipeline(model=model, **kwargs)
        score = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
        print(f'{name:35} : {score.mean()}')
score(X=frame, y=y, feature='Raw')
score(X=frame, y=y, feature='Processed', topK=10000)
score(X=frame, y=y, feature='Lemmatized', topK=10000)
score(X=frame, y=y, feature='Stemming', topK=8000)
_input0.head()
_input0['Processed'] = process_data(_input0['text'])
_input0['Lemmatized'] = _input0['Processed'].apply(lemmatize)
_input0['Stemming'] = _input0['Processed'].apply(stemming)
_input0.head()
feature = 'Lemmatized'
model = ComplementNB()
vectors = count_vectorizer.fit_transform(frame[feature])