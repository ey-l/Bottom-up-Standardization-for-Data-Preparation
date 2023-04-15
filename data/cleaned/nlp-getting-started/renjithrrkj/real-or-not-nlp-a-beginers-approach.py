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
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.head()
print('the training dataset has', train.shape[0], 'rows', train.shape[1], 'columns')
plt.style.use('seaborn')
p1 = sns.countplot(x='target', data=train)
for p in p1.patches:
    p1.annotate('{:6.2f}%'.format(p.get_height() / len(train) * 100), (p.get_x() + 0.1, p.get_height() + 50))
plt.gca().set_ylabel('samples')

def cleaned(text):
    text = re.sub('\\n', '', text)
    text = text.lower()
    text = re.sub('\\d', '', text)
    text = re.sub('[^\\x00-\\x7f]', ' ', text)
    text = re.sub('[^\\w\\s]', '', text)
    text = re.sub('http\\S+|www.\\S+', '', text)
    return text
train['cleaned'] = train['text'].apply(lambda x: cleaned(x))
test['cleaned'] = test['text'].apply(lambda x: cleaned(x))
train.head()
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
X = train['cleaned'].to_numpy()
y = train['target'].to_numpy()
for (train_index, test_index) in sss.split(X, y):
    print('TRAIN:', train_index, 'TEST:', test_index)
    (X_train, X_test) = (X[train_index], X[test_index])
    (y_train, y_test) = (y[train_index], y[test_index])
tweets_pipeline = Pipeline([('CVec', CountVectorizer(stop_words='english')), ('Tfidf', TfidfTransformer())])
X_train_transformed = tweets_pipeline.fit_transform(X_train)
X_test_transformed = tweets_pipeline.transform(X_test)
SVC_clf = SVC()