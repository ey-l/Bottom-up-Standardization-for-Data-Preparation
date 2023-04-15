import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
df = pd.read_csv('data/input/nlp-getting-started/train.csv')
df.shape
df.sample(2)
sns.histplot(df['target'])
df = df.drop(labels=['id', 'location'], axis=1)
df.isna().sum()
df['text'].duplicated().sum()
df = df.drop_duplicates(subset=['text'], keep=False)
df = df.dropna()
y_df = df['target']
df = df.drop(labels=['target'], axis=1)
df = df.reset_index()
df = df.drop(labels=['index'], axis=1)
df['keyword'] = df['keyword'].str.replace('%20', ' ')
df['keyword'].sample(50)
keyword_dummies = pd.get_dummies(df['keyword'])
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english')
text_vectorizer = vect.fit_transform(df['text'])
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
text_vectorizer2 = vect2.fit_transform(df['text'])
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