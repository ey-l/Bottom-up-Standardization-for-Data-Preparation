import pandas as pd
import numpy as np
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
tfidf_vetorizer = feature_extraction.text.TfidfVectorizer()
train_vectors = tfidf_vetorizer.fit_transform(_input1['text'])
test_vectors = tfidf_vetorizer.transform(_input0['text'])
print(train_vectors)
model = linear_model.RidgeClassifier()