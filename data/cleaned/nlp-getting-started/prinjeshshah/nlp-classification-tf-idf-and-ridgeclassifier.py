import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_df = train_df.drop('id', axis=1)
train_df = train_df.drop('keyword', axis=1)
train_df = train_df.drop('location', axis=1)
test_df = test_df.drop('id', axis=1)
test_df = test_df.drop('keyword', axis=1)
test_df = test_df.drop('location', axis=1)
count_vectorizer = feature_extraction.text.TfidfVectorizer()
train_vectors = count_vectorizer.fit_transform(train_df['text'])
test_vectors = count_vectorizer.transform(test_df['text'])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df['target'], cv=3, scoring='f1')
scores