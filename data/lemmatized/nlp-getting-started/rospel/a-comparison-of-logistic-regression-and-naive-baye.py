import pandas as pd
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
_input1 = _input1.drop(['keyword', 'location', 'id'], axis=1, inplace=False)
test_ids = _input0.id.values
_input0 = _input0.drop(['keyword', 'location', 'id'], axis=1, inplace=False)
test_ids
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5, 5))
labels = ('Non-disaster', 'Disaster')
counts = _input1.target.value_counts().values
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#7DB9B6', '#E96479'])
plt.axis('equal')
plt.title('Tweet Class Distribution')
import numpy as np
disaster = _input1['text'].loc[_input1.target == 1].values
non_disaster = _input1['text'].loc[_input1.target == 0].values
pos_sample = np.random.randint(0, len(disaster))
neg_sample = np.random.randint(0, len(non_disaster))
print('\x1b[92m' + 'Non-disaster tweet: ' + non_disaster[neg_sample])
print('\x1b[91m' + 'Disaster tweet: ' + disaster[pos_sample])
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from nltk.tokenize import TweetTokenizer
import string
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stopwords_english = stopwords.words('english')
stemmer = PorterStemmer()

def process_tweets(tweets, proccesor):
    clean_tweets = []
    preproces_tweets = []
    if proccesor == 'emojis':
        for x in tweets:
            x = re.sub('^RT[\\s]+', '', x)
            x = re.sub('https?://[^\\s\\n\\r]+', '', x)
            x = re.sub('#', '', x)
            clean_tweets.append(tokenizer.tokenize(x))
        for y in clean_tweets:
            preproces_tweet = []
            for word in y:
                if word not in stopwords_english and word not in string.punctuation:
                    word = stemmer.stem(word)
                    preproces_tweet.append(word)
            preproces_tweets.append(preproces_tweet)
    elif proccesor == 'no_emojis':
        for x in tweets:
            x = re.sub('^RT[\\s]+', '', x)
            x = re.sub('https?://[^\\s\\n\\r]+', '', x)
            x = re.sub('#', '', x)
            x = re.sub('[^a-zA-Z0-9]+', ' ', x)
            clean_tweets.append(tokenizer.tokenize(x))
        for y in clean_tweets:
            preproces_tweet = []
            for word in y:
                if word not in stopwords_english and word not in string.punctuation:
                    word = stemmer.stem(word)
                    preproces_tweet.append(word)
            preproces_tweets.append(preproces_tweet)
    return preproces_tweets
train_tweets_withe = process_tweets(_input1.text.values, 'emojis')
train_tweets_oute = process_tweets(_input1.text.values, 'no_emojis')
rand_value = np.random.randint(0, len(_input1.text))
print('Original: ', _input1.text.values[rand_value], '\n With Emojis: ', train_tweets_withe[rand_value], '\n Without Emojis: ', train_tweets_oute[rand_value])

def build_freqs(tweets, labels):
    label_list = np.squeeze(labels).tolist()
    freqs = {}
    for (y, tweet) in zip(label_list, tweets):
        for word in tweet:
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs
targets = list(_input1.target.values)
freq_dict_withe = build_freqs(train_tweets_withe, targets)
freq_dict_oute = build_freqs(train_tweets_oute, targets)
word = stemmer.stem('earthquak')
word
keys = ['earthquak', ':(', ':)', 'tornado', 'music', 'colour', 'happi', 'sad', 'so', 'fun', 'peopl']
data = []
for word in keys:
    pos = 0
    neg = 0
    if (word, 1) in freq_dict_withe:
        pos = freq_dict_withe[word, 1]
    if (word, 0) in freq_dict_withe:
        neg = freq_dict_withe[word, 0]
    data.append([word, pos, neg])
(fig, ax) = plt.subplots(figsize=(8, 8))
x = np.log([x[1] + 1 for x in data])
y = np.log([x[2] + 1 for x in data])
ax.scatter(x, y, c='#23343A', label='Words')
plt.xlabel('Log Disaster count')
plt.ylabel('Log non disaster count')
plt.title('Word count separation')
for i in range(0, len(data)):
    ax.annotate(data[i][0], (x[i], y[i]), fontsize=12)
ax.plot([0, 9], [0, 9], color='#C13F2A', label='Separation')
plt.grid()

def extract_features(tweet, freqs):
    x = np.zeros((1, 3))
    x[0, 0] = 1
    for word in tweet:
        x[0, 1] += freqs.get((word, 1), 0)
        x[0, 2] += freqs.get((word, 0), 0)
    return x
train_tweets_withe = [extract_features(x, freq_dict_withe) for x in train_tweets_withe]
train_tweets_oute = [extract_features(x, freq_dict_oute) for x in train_tweets_oute]
print(train_tweets_withe[10])
print(train_tweets_oute[10])
from sklearn.model_selection import train_test_split
(X_train01, X_val01, y_train01, y_val01) = train_test_split(train_tweets_withe, targets, stratify=targets, train_size=0.9, shuffle=True, random_state=0)
(X_train02, X_val02, y_train02, y_val02) = train_test_split(train_tweets_oute, targets, stratify=targets, train_size=0.9, shuffle=True, random_state=0)
X_train01 = np.concatenate(X_train01, axis=0)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
clf_withe = LogisticRegression(random_state=0, solver='liblinear', max_iter=100, verbose=1, n_jobs=1)