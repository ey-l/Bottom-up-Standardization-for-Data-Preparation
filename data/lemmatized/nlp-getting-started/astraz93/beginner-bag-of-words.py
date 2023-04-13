import pandas as pd
text = ['This is the first document.', 'This document is the second document.', 'And this is the third one.', 'Is this the first document?']
print(text)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text)
columns = vectorizer.get_feature_names_out()
df = pd.DataFrame(X.toarray(), columns=columns, index=text)
df
text = ['The office building is open today']
vectorizer = CountVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(text)
columns = vectorizer.get_feature_names_out()
df = pd.DataFrame(X.toarray(), columns=columns, index=text)
df
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(text)
columns = vectorizer.get_feature_names_out()
df = pd.DataFrame(X.toarray(), columns=columns, index=text)
df
vectorizer = CountVectorizer(ngram_range=(3, 3))
X = vectorizer.fit_transform(text)
columns = vectorizer.get_feature_names_out()
df = pd.DataFrame(X.toarray(), columns=columns, index=text)
df