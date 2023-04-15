import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('data/input/nlp-getting-started/train.csv')
df.head(5)
df.info()
sns.heatmap(df.isnull())
df.drop(['location', 'keyword'], axis=1, inplace=True)
df
real = df[df['target'] == 1]
real
unreal = df[df['target'] == 0]
unreal
print('real disaster message percentage:', len(real) / len(df) * 100)
print('fake disaster message percentage:', len(unreal) / len(df) * 100)
sns.countplot(df['target'])
import string
string.punctuation
from nltk.corpus import stopwords
stopwords.words('english')

def message_cleaning(message):
    test_punc_removed = [char for char in message if char not in string.punctuation]
    test_punc_removed_joined = ''.join(test_punc_removed)
    test_punc_removed_joined_clean = [word for word in test_punc_removed_joined.split(' ') if word not in stopwords.words('english')]
    return test_punc_removed_joined_clean
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer=message_cleaning)
disaster_tweet_vectorizer = vectorizer.fit_transform(df['text'])
print(vectorizer.get_feature_names())
print(disaster_tweet_vectorizer.toarray())
disaster_tweet_vectorizer.shape
label = df['target']
label.shape
X = disaster_tweet_vectorizer
X = X.toarray()
X
y = label
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
LR = LogisticRegression()
DTC = DecisionTreeClassifier()
RFC = RandomForestClassifier()
NB = GaussianNB()