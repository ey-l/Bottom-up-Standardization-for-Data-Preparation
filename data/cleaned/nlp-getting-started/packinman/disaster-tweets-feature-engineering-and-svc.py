import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
train_df.head()
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
test_df.head()
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
word_1 = '  '.join(list(train_df[train_df['target'] == 1]['text']))
word_1 = WordCloud(width=600, height=500).generate(word_1)
plt.figure(figsize=(13, 9))
plt.imshow(word_1)

word_0 = '  '.join(list(train_df[train_df['target'] == 0]['text']))
word_0 = WordCloud(width=600, height=500).generate(word_0)
plt.figure(figsize=(13, 9))
plt.imshow(word_0)

train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)
miss_per = train_df.isnull().sum() / len(train_df) * 100
miss_per = miss_per.sort_values(ascending=False)
sns.barplot(x=miss_per.index, y=miss_per)
plt.xlabel('Features')
plt.ylabel('% of Missing Values')

train_df['location'] = train_df['location'].fillna('None')
train_df['keyword'] = train_df['keyword'].fillna('None')
test_df['location'] = test_df['location'].fillna('None')
test_df['keyword'] = test_df['keyword'].fillna('None')
import nltk
from nltk import TweetTokenizer
tokenizer = TweetTokenizer()
train_df['tokens'] = [tokenizer.tokenize(item) for item in train_df.text]
test_df['tokens'] = [tokenizer.tokenize(item) for item in test_df.text]
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_item(item):
    new_item = []
    for x in item:
        x = lemmatizer.lemmatize(x)
        new_item.append(x)
    return ' '.join(new_item)
train_df['tokens'] = [lemmatize_item(item) for item in train_df.tokens]
test_df['tokens'] = [lemmatize_item(item) for item in test_df.tokens]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
target = train_df['target']
train_df = train_df.drop('target', axis=1)
train_x_vec = vectorizer.fit_transform(train_df.tokens)
test_x_vec = vectorizer.transform(test_df.tokens)
X = train_x_vec
y = target
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.svm import SVC
class_svc = SVC(probability=True, random_state=0)