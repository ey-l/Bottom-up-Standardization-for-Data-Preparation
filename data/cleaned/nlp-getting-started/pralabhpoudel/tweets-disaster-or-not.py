import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import TweetTokenizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('data/input/nlp-getting-started/train.csv')
train_data.head()
train_data.info()
train_data.describe()
Miss_Percent = 100 * (train_data.isnull().sum() / len(train_data))
Miss_Percent = Miss_Percent[Miss_Percent > 0].sort_values(ascending=False).round(1)
DataFrame = pd.DataFrame(Miss_Percent)
miss_percent_table = DataFrame.rename(columns={0: '% of Missing Values'})
MissPercent = miss_percent_table
MissPercent
train_data['keyword'].unique()
train_data['keyword'] = train_data['keyword'].fillna('None')
train_data['location'].unique()
train_data['location'] = train_data['location'].fillna('Unavailable')
train_data = train_data.drop('id', axis=1)
train_data.head()
sns.countplot(data=train_data, x='target')
data_disaster_nl = ' '.join(list(train_data[train_data['target'] == 1]['text']))
data_wc_nl = WordCloud(width=600, height=512).generate(data_disaster_nl)
plt.figure(figsize=(13, 9))
plt.imshow(data_wc_nl)

data_non_disaster_nl = ' '.join(list(train_data[train_data['target'] == 0]['text']))
data_wc_nd_nl = WordCloud(width=600, height=512).generate(data_non_disaster_nl)
plt.figure(figsize=(13, 9))
plt.imshow(data_wc_nd_nl)

tokenizer = TweetTokenizer()
tokens = [tokenizer.tokenize(word) for word in train_data['text']]
train_data = train_data.assign(tokens=tokens)
train_data
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def lemmatize_stem_item(item):
    new_item = []
    for x in item:
        x = lemmatizer.lemmatize(x)
        x = stemmer.stem(x)
        new_item.append(x)
    return ' '.join(new_item)
if not 'stemmed' in train_data:
    train_data.tokens = [lemmatize_stem_item(item) for item in train_data.tokens]
    train_data['stemmed'] = True
train_data
vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_data.tokens)
train_x_vectors
X = train_x_vectors
y = train_data['target']
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.25, random_state=42)
print('Number of rows in training set: ' + str(X_train.shape))
print('Number of rows in validation set: ' + str(X_val.shape))
from sklearn.linear_model import LogisticRegression