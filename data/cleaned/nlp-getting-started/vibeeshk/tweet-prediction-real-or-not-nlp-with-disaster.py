import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
filepath = 'data/input/nlp-getting-started/train.csv'
traindata = pd.read_csv(filepath)
filepaths = 'data/input/nlp-getting-started/test.csv'
testdata = pd.read_csv(filepaths)
traindata.head()
import pandas as pd
import nltk
import numpy as np
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from nltk.corpus import words
word_list = words.words()
traindata2 = traindata
testdata2 = testdata
added_df = pd.concat([traindata, testdata])
len(added_df['keyword'])
textlist = added_df['text'].tolist()
keywordlist = added_df['keyword'].tolist()
targetlist = added_df['target'].tolist()
len(textlist)
stop_words = set(stopwords.words('english'))
sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words
textlist2 = []
for i1 in textlist:
    i1 = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', i1, flags=re.MULTILINE)
    i1 = re.sub('[0-9]', '', i1)
    i1 = re.sub('^https?:\\/\\/.*[\\r\\n]*', '', i1, flags=re.MULTILINE)
    i1 = re.sub('[^\\w\\s]', '', i1)
    i1 = re.sub('\\W', ' ', i1)
    i1 = i1.lower()
    i2 = word_tokenize(i1)
    i2 = [word for word in i2 if not word in all_stopwords]
    i2 = [w for w in i2 if not w in stop_words]
    i3 = ' '.join(i2)
    textlist2.append(i3)
lemmatizer = WordNetLemmatizer()
textlist3 = []
for i4 in textlist2:
    i4 = lemmatizer.lemmatize(i4)
    textlist3.append(i4)
len(textlist3)
added_df = added_df.drop('text', axis=1)
added_df['text'] = textlist3
cv = CountVectorizer()
added_df = cv.fit_transform(added_df['text'])
added_df = pd.DataFrame(added_df.todense())
added_df['keywords'] = keywordlist
added_df['target'] = targetlist
added_df['keywords'] = added_df['keywords'].replace(np.nan, 'keywords', regex=True)
added_df['keywords'] = 'keywords-' + added_df['keywords'].astype(str)
added_df.head()
one_hot = pd.get_dummies(added_df['keywords'])
added_df = added_df.drop('keywords', axis=1)
added_df = added_df.join(one_hot)
added_df.head()
added_df = added_df.replace(np.nan, 0, regex=True)
added_df = added_df.reset_index()
training = added_df.head(7613)
testing = added_df.iloc[7613:]
y = training['target']
x = training.drop('target', axis=1)
testing = testing.drop('target', axis=1)
LogisticRegressor = LogisticRegression(max_iter=10000)