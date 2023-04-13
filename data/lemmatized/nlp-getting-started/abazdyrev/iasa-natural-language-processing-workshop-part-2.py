import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv', index_col='id')
_input1 = _input1.drop(columns=['keyword', 'location'])
(X, y) = (_input1['text'], _input1['target'].values)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
word_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 3), lowercase=True, min_df=5, max_features=30000)
char_vectorizer = TfidfVectorizer(analyzer='char', stop_words='english', ngram_range=(3, 6), lowercase=True, min_df=5, max_features=50000)
vectorizer = FeatureUnion([('word_vectorizer', word_vectorizer), ('char_vectorizer', char_vectorizer)])