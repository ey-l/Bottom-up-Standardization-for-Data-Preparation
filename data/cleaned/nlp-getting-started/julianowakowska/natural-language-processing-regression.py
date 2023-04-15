import pandas as pd
import numpy as np
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.head()
tfidf_vetorizer = feature_extraction.text.TfidfVectorizer()
train_vectors = tfidf_vetorizer.fit_transform(train['text'])
test_vectors = tfidf_vetorizer.transform(test['text'])
print(train_vectors)
model = linear_model.RidgeClassifier()