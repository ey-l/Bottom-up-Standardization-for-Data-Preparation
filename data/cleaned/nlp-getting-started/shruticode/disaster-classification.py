import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv', dtype={'id': np.int16, 'target': np.int8})
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv', dtype={'id': np.int16})
train_df.head(1)
train_df[train_df['target'] == 0]['text'].values[3]
train_df[train_df['target'] == 1]['text'].head(2)
print(train_df.shape)
print(test_df.shape)
print(f"Number of unique values in keyword = {train_df['keyword'].nunique()} (Training) - {test_df['keyword'].nunique()} (Test)")
print(f"Number of unique values in location = {train_df['location'].nunique()} (Training) - {test_df['location'].nunique()} (Test)")
train_df['org_text'] = train_df['text']
test_df['org_text'] = test_df['text']
train_df['text'] = train_df['text'].apply(lambda x: x.lower())
test_df['text'] = test_df['text'].apply(lambda x: x.lower())
from gensim.utils import tokenize
train_df['text'] = train_df['text'].apply(lambda x: list(tokenize(x)))
test_df['text'] = test_df['text'].apply(lambda x: list(tokenize(x)))
import gensim.parsing.preprocessing
stopwords = gensim.parsing.preprocessing.STOPWORDS
train_df['text'] = train_df['text'].apply(lambda x: [item for item in x if item not in stopwords])
test_df['text'] = test_df['text'].apply(lambda x: [item for item in x if item not in stopwords])
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
train_df['text'] = train_df['text'].apply(lambda x: [stemmer.stem(item) for item in x])
test_df['text'] = test_df['text'].apply(lambda x: [stemmer.stem(item) for item in x])
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
train_df['text'] = train_df['text'].apply(lambda x: lemma.lemmatize(' '.join(x)))
test_df['text'] = test_df['text'].apply(lambda x: lemma.lemmatize(' '.join(x)))
count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train_df['text'])
test_vectors = count_vectorizer.transform(test_df['text'])
classifier = linear_model.RidgeClassifier()
score = model_selection.cross_val_score(classifier, train_vectors, train_df['target'], cv=3, scoring='f1')
score