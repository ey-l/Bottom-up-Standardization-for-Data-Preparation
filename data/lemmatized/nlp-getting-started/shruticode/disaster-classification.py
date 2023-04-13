import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv', dtype={'id': np.int16, 'target': np.int8})
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv', dtype={'id': np.int16})
_input1.head(1)
_input1[_input1['target'] == 0]['text'].values[3]
_input1[_input1['target'] == 1]['text'].head(2)
print(_input1.shape)
print(_input0.shape)
print(f"Number of unique values in keyword = {_input1['keyword'].nunique()} (Training) - {_input0['keyword'].nunique()} (Test)")
print(f"Number of unique values in location = {_input1['location'].nunique()} (Training) - {_input0['location'].nunique()} (Test)")
_input1['org_text'] = _input1['text']
_input0['org_text'] = _input0['text']
_input1['text'] = _input1['text'].apply(lambda x: x.lower())
_input0['text'] = _input0['text'].apply(lambda x: x.lower())
from gensim.utils import tokenize
_input1['text'] = _input1['text'].apply(lambda x: list(tokenize(x)))
_input0['text'] = _input0['text'].apply(lambda x: list(tokenize(x)))
import gensim.parsing.preprocessing
stopwords = gensim.parsing.preprocessing.STOPWORDS
_input1['text'] = _input1['text'].apply(lambda x: [item for item in x if item not in stopwords])
_input0['text'] = _input0['text'].apply(lambda x: [item for item in x if item not in stopwords])
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
_input1['text'] = _input1['text'].apply(lambda x: [stemmer.stem(item) for item in x])
_input0['text'] = _input0['text'].apply(lambda x: [stemmer.stem(item) for item in x])
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
_input1['text'] = _input1['text'].apply(lambda x: lemma.lemmatize(' '.join(x)))
_input0['text'] = _input0['text'].apply(lambda x: lemma.lemmatize(' '.join(x)))
count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(_input1['text'])
test_vectors = count_vectorizer.transform(_input0['text'])
classifier = linear_model.RidgeClassifier()
score = model_selection.cross_val_score(classifier, train_vectors, _input1['target'], cv=3, scoring='f1')
score