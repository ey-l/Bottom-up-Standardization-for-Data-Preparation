import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import nltk
from nltk.corpus import stopwords
import textblob
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
train = train.drop(['location'], axis=1)
train.head()
train.shape
train.isnull().sum()
train['text'] = train['text'].apply(lambda x: ' '.join((x.lower() for x in x.split())))
train['keyword'] = train['keyword'].str.replace('\\d', '')
train['keyword'] = train['keyword'].str.replace('[^\\w\\s]', '')
keyword = pd.unique(train.keyword)[1:]
keys = []
for text in train.text:
    t_k = []
    for key in keyword:
        if text.find(str(key)) != -1:
            t_k.append(key)
        else:
            continue
    keys.append(t_k)
train['keyword'] = keys
df = train.copy()
df['text'] = df['text'].apply(lambda x: ' '.join((x.lower() for x in x.split())))
df['text'] = df['text'].str.replace('[^\\w\\s]', '')
df['keyword'] = df['keyword'].apply(lambda x: str(x))
df['keyword'] = df['keyword'].str.replace('[^\\w\\s]', '')
df['text'] = df['text'].str.replace('\\d', '')
df['keyword'] = df['keyword'].str.replace('\\d', '')
import nltk
from nltk.corpus import stopwords
sw = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: ' '.join((x for x in x.split() if x not in sw)))
del_word = pd.Series(' '.join(df['text']).split()).value_counts()[1500:]
df['text'] = df['text'].apply(lambda x: ' '.join((x for x in x.split() if x not in del_word)))
from textblob import Word
df['text'] = df['text'].apply(lambda x: ' '.join([Word(word).lemmatize() for word in x.split()]))
features = keyword
counts = []
for i in features:
    counts.append(df['text'].apply(lambda x: len([x for x in x.split() if x.startswith(i)])).sum())
df_fre = pd.DataFrame(columns=['names', 'frequency'])
df_fre['names'] = features
df_fre['frequency'] = counts
df_fre.head()
import plotly.express as px
df_class = pd.value_counts(df['target'], sort=True).sort_index()
fig = px.bar(df_class)
fig.show()
import plotly.express as px
fig = px.bar(df_fre, x=df_fre.names, y=df_fre.frequency, title='Frequency of Keyword')
fig.show()
import plotly.express as px
fig = px.pie(df_fre, values='frequency', names='names', title='Population of Keywords')
fig.update_traces(textposition='inside')
fig.update_layout(uniformtext_minsize=7, uniformtext_mode='hide')
fig.show()
words = ' '.join((x for x in df.keyword))
plt.figure(figsize=(20, 10))
wordcloud = WordCloud(background_color='white', width=2000, height=800).generate(words)
plt.imshow(wordcloud)
plt.axis('off')

words = ' '.join((x for x in df.text))
plt.figure(figsize=(20, 10))
wordcloud = WordCloud(background_color='white', width=2000, height=800).generate(words)
plt.imshow(wordcloud)
plt.axis('off')

from sklearn import model_selection, preprocessing
(x_train, x_test, y_train, y_test) = model_selection.train_test_split(df['text'], df['target'], random_state=21, test_size=0.33)
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=500)