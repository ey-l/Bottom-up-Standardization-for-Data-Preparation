import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
print('Train data shape is:', train.shape)
print('Test data shape is:', test.shape)
train.head(3)
test.head(3)
ax = sns.countplot(x='target', data=train)
target_0 = train[train['target'] == 0]['text']
print(target_0.values[6])
target_1 = train[train['target'] == 1]['text']
print(target_1.values[3])
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 5))
tweet_len = target_1.str.len()
ax1.hist(tweet_len, color='red')
ax1.set_title('Disaster tweets')
tweet_len = target_0.str.len()
ax2.hist(tweet_len, color='blue')
ax2.set_title('Not disaster tweets')
fig.suptitle('Length of tweets')

top_location = train['location'].value_counts().reset_index()
top10_location = top_location.iloc[:10]
top10_location
ax = sns.barplot(x='index', y='location', data=top10_location)
plt.xticks(rotation=45)
from wordcloud import WordCloud
word_freq_dic = train[train.target == 1].keyword.value_counts().to_dict()
wordcloud = WordCloud(background_color='white', colormap='autumn', width=700, height=700, random_state=42).generate_from_frequencies(word_freq_dic)
plt.figure(figsize=(6, 6))
plt.imshow(wordcloud)
plt.title('Word Frequency', size=13)
plt.axis('off')

count_vectorizer = feature_extraction.text.CountVectorizer()
example_train_vectors1 = count_vectorizer.fit_transform(train['text'][0:5])
example_train_vectors1
Tfidf_vectorizer = feature_extraction.text.TfidfVectorizer()
example_train_vectors2 = Tfidf_vectorizer.fit_transform(train['text'][0:5])
print(example_train_vectors1[0].todense().shape)
print(example_train_vectors1[0].todense())
print(example_train_vectors2[0].todense().shape)
print(example_train_vectors2[0].todense())
train_vectors_cv = count_vectorizer.fit_transform(train['text'])
test_vectors_cv = count_vectorizer.transform(test['text'])
train_vectors_tv = Tfidf_vectorizer.fit_transform(train['text'])
test_vectors_tv = Tfidf_vectorizer.transform(test['text'])
test_vectors_cv
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors_tv, train['target'], cv=3, scoring='f1')
scores