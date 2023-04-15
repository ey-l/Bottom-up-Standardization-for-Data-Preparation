import pandas as pd
import math
from sklearn.feature_extraction.text import HashingVectorizer
text = ['The sky is blue and beautiful', 'The king is old and the queen is beautiful', 'Love this beautiful blue sky', 'The beautiful queen and the old king']
vectorizer = HashingVectorizer(n_features=8, norm=None, stop_words='english')
X = vectorizer.fit_transform(text)
matrix = pd.DataFrame(X.toarray())
matrix
vectorizer = HashingVectorizer(n_features=5, norm=None, stop_words='english')
X = vectorizer.fit_transform(text)
matrix = pd.DataFrame(X.toarray())
matrix