import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.shape
_input1.sample(2)
sns.histplot(_input1['target'])
_input1 = _input1.drop(labels=['id', 'location'], axis=1)
_input1.isna().sum()
_input1['text'].duplicated().sum()
_input1 = _input1.drop_duplicates(subset=['text'], keep=False)
_input1 = _input1.dropna()
y_df = _input1['target']
_input1 = _input1.drop(labels=['target'], axis=1)
_input1 = _input1.reset_index()
_input1 = _input1.drop(labels=['index'], axis=1)
_input1['keyword'] = _input1['keyword'].str.replace('%20', ' ')
_input1['keyword'].sample(50)
keyword_dummies = pd.get_dummies(_input1['keyword'])
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english')
text_vectorizer = vect.fit_transform(_input1['text'])
arr = vect.get_feature_names_out()
forbidden = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_', '@', 'ü', 'ò', 'û', '$', '%', '*', '#', '/', 'â', '!', '[', ']', '.', ';', 'ç', '>', 'å']
stop = []
i = -1
for item in arr:
    i = i + 1
    for letter in item:
        if letter in forbidden:
            stop.append(item)
print(len(stop))
stop = set(stop)
arr = set(arr)
vocab = arr - stop
len(vocab)
from sklearn.feature_extraction.text import CountVectorizer
vect2 = CountVectorizer(stop_words='english', vocabulary=vocab)
text_vectorizer2 = vect2.fit_transform(_input1['text'])
arr2 = vect2.get_feature_names()
text_df = pd.DataFrame(text_vectorizer2.toarray())
text_df
proc_df = pd.concat([keyword_dummies, text_df], axis=1)
proc_df
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(proc_df, y_df)
(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB(var_smoothing=0.01873817422860384)