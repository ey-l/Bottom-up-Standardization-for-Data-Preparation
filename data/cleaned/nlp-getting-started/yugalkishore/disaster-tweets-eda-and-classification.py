import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
df = pd.read_csv('data/input/nlp-getting-started/train.csv')
df.info()
df.head()
df['msg_len'] = df['text'].apply(len)
df['msg_len']
df['msg_len'].describe()
df[df['msg_len'] == 157]['text'].iloc[0]
sns.barplot(x=df['target'], y=df['msg_len'], data=df)
sns.set_style('darkgrid')
df.hist(column='msg_len', by='target', sharey=True, bins=50, figsize=(12, 4))
df.drop(['id', 'keyword', 'location', 'msg_len'], axis=1, inplace=True)
import string
import time
from nltk.corpus import stopwords
commonwords = stopwords.words('english')
from nltk.corpus import words
import re
from nltk.stem.porter import PorterStemmer
string.punctuation
corpus = []
for i in range(0, df.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = ' '.join(review)
    corpus.append(review)
time.sleep(3)
corpus[0:5]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
nb_model = MultinomialNB()