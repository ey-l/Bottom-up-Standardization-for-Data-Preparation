import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)
train_data = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_data.head()
test_data.head()
text = train_data['text']
text.str.len().hist()
import matplotlib.pyplot as plt
arr = []
for i in text:
    tmp = i.split(' ')
    arr.append(len(tmp))
plt.grid()
plt.hist(arr)

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = set(stopwords.words('english'))
import numpy as np
corpus = []
tmp = []
for i in text:
    tmp.append(i.split())
for i in tmp:
    for words in i:
        corpus.append(words)
from collections import defaultdict
dic = defaultdict(int)
for i in corpus:
    if i in stop:
        dic[i] = dic[i] + 1
val = dic.values()
val = sorted(val, reverse=True)
val = val[0:11]
d = {}
for i in list(dic.keys()):
    if dic[i] in val:
        d[i] = dic[i]
plt.bar(list(d.keys()), list(d.values()))
from nltk.util import ngrams
import collections
tex = ''
for i in text:
    tex = tex + str(i.strip('[]'))
tok = tex.split()
b_grams = ngrams(tok, 2)
t_grams = ngrams(tok, 3)
bigrams = collections.Counter(b_grams)
trigrams = collections.Counter(t_grams)
bi = bigrams.most_common(10)
ti = trigrams.most_common(10)

def plot_ngrams(bi):
    word = []
    idx = []
    for i in range(len(bi)):
        word.append(str(bi[i][0]))
        idx.append(bi[i][1])
    plt.figure(figsize=(10, 8))
    plt.bar(word, idx)
    plt.xticks(word, word, rotation='vertical')
    plt.xlabel('Ngrams')
    plt.ylabel('Frequency')
    plt.title('Ngram with Frequency')

plot_ngrams(bi)
plot_ngrams(ti)
import re
a = ''
for i in range(len(text)):
    tmp = re.search('#[a-zA-Z0-9]*', text[i])
    if tmp is not None:
        tm = re.sub('#', '', tmp.group())
        a = a + str(tm) + ' '
    else:
        a = a + ''
from wordcloud import WordCloud, STOPWORDS
stopwords = train_data['keyword'].unique()
wordcloud = WordCloud()
wordcloud = wordcloud.generate(a)
fig = plt.figure(1, figsize=(12, 12))
plt.axis('off')
plt.imshow(wordcloud)

from textblob import TextBlob
po = []
for i in text:
    analysis = TextBlob(i)
    po.append(analysis.sentiment.polarity)
dic = {'polarity': po, 'Tweets': text}
plt.plot(po[:20])