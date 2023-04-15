import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from sklearn import feature_extraction, model_selection, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
import nltk
nltk.download('stopwords')
from nltk import word_tokenize
import warnings
warnings.filterwarnings('ignore')
import wordcloud
from wordcloud import WordCloud, STOPWORDS
import re
import matplotlib.pyplot as plt
train_path = 'data/input/nlp-getting-started/train.csv'
test_path = 'data/input/nlp-getting-started/test.csv'
df = pd.read_csv(train_path, index_col='id')
df_t = pd.read_csv(test_path, index_col='id')
df.head()
tweets = ''
stopwords = set(STOPWORDS)
for val in df.text:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
        tokens[i] = re.sub('[/|$|.|!|<|>]+#?|-', '', tokens[i])
        tokens[i] = re.sub('^http?:\\/\\/.*[\\r\\n]*', '', tokens[i], flags=re.MULTILINE)
    tweets += ' '.join(tokens) + ' '
wordcloud = WordCloud(width=800, height=800, scale=10, stopwords=stopwords, min_font_size=10, max_font_size=100, max_words=100, background_color='black').generate(tweets)
print(tokens)
plt.figure(figsize=(12, 10), facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)

df[df['target'] == 0]['text'].values[0]
df[df['target'] == 1]['text'].values[0]
count_vectorizer = feature_extraction.text.CountVectorizer()
sample_train_vector = count_vectorizer.fit_transform(df['text'][0:5])
for i in range(5):
    print(sample_train_vector[i].todense().shape)
    print(sample_train_vector[i].todense())
train_vectors = count_vectorizer.fit_transform(df['text'])
print(train_vectors.shape)
test_vectors = count_vectorizer.transform(df_t['text'])
clf = linear_model.RidgeClassifier()
clf1 = LogisticRegression(max_iter=750)
clf2 = RandomForestClassifier(n_estimators=500)
clf3 = MLPClassifier(hidden_layer_sizes=(200, 100, 50, 25), max_iter=200)
clf6 = KNeighborsClassifier(n_neighbors=7)
params = {'max_iter': (2000, 2500, 3000), 'solver': ('lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'), 'fit_intercept': (True, False), 'penalty': ('l2', 'l1', 'elasticnet')}
clf4 = GridSearchCV(estimator=LogisticRegression(max_iter=750), param_grid=params, cv=3)