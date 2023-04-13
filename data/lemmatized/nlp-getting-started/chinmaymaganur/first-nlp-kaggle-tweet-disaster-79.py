import numpy as np
import pandas as pd
import seaborn as sns
import plotly
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import stopwords
import nltk
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
print('train', _input1.shape)
print('test', _input1.shape)
_input1.head()
_input1['text'][1000]
print('train,null%\n', _input1.isnull().mean())
print()
print('test,null%\n', _input0.isnull().mean())
disaster = _input1[_input1['target'] == 1]
non_dist = _input1[_input1['target'] == 0]
d = disaster['keyword'].value_counts()[:20]
nd = non_dist['keyword'].value_counts()[:20]
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=2)
fig.add_traces(go.Bar(y=d.index, x=d.values, orientation='h', name='disaster keywords'), 1, 1)
fig.add_traces(go.Bar(y=nd.index, x=nd.values, orientation='h', name='non_dist kewords'), 1, 2)
loc_d = disaster['location'].value_counts()[:10]
loc_nd = non_dist['location'].value_counts()[:10]
fig = make_subplots(rows=1, cols=2)
fig.add_traces(go.Bar(y=loc_d.index, x=loc_d.values, orientation='h', name='Most disaster loc'), 1, 1)
fig.add_traces(go.Bar(y=loc_nd.index, x=loc_nd.values, orientation='h', name='non_dist Loc '), 1, 2)
import plotly.figure_factory as ff
d1 = disaster['text'].apply(len)
nd1 = non_dist['text'].apply(len)
ff.create_distplot([d1, nd1], ['len_d1', 'len_nd1'])
print('mean_len_d1', d1.mean())
print('mean_len_nd1', nd1.mean())
non_dist['text'].str.len().sort_values(ascending=False)[:10]
print(non_dist['text'][1270])
print(non_dist['text'][4801])
print(non_dist['text'][261])
print(non_dist['text'][5379])
disaster['text'].str.len().sort_values(ascending=False)[:10]
print(disaster['text'][614])
print(disaster['text'][635])
print(disaster['text'][2718])
print(disaster['text'][1111])
import string
import re
"\nimport re\na='http://t.co/FYJWjDkM5I this is it'\na=re.sub('http://[a-z]+\\.[a-z]+/[a-zA-Z]+','',non_dist['text'][6555],)\n#re.findall('http://[a-z]+\\.[a-z]+/[a-zA-Z]+',a)\n\nre.sub('[^\\w]',' ',a)\n"
'a=\'@IcyMagistrate \x89ÛÓher upper arm\x89ÛÒ those /friggin/ icicle projectiles\x89ÛÒ and leg from various other wounds the girl looks like a miniature more\x89ÛÓ\'\na.encode("ascii", errors="ignore").decode()\n#output==\'@IcyMagistrate her upper arm those /friggin/ icicle projectiles and leg from various other wounds the girl looks like a miniature\'http.?://[a-z]+\\.[a-z]+/[a-zA-Z0-9]+ more\'\nhttp.?://[a-z]+\\.[a-z]+/[a-zA-Z0-9]+'

def clean_text(text):
    emoji_pattern = re.compile('[😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿✂-➰Ⓜ-🉑]+', flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    text = text.encode('ascii', errors='ignore').decode()
    text = text.lower()
    text = re.sub('http.?://[a-z]+\\.[a-z]+/[a-zA-Z0-9]+', '', text)
    text = re.sub('&amp', ' and', text)
    text = re.sub('gt', 'greater than', text)
    text = re.sub('lt', 'lesser than', text)
    text = re.sub('rt', 'retweet', text)
    text = re.sub("n't", ' not', text)
    text = re.sub("'s", ' is', text)
    text = re.sub("'ll", 'will', text)
    text = re.sub("'ve", 'have', text)
    text = re.sub("i'm", 'i am', text)
    text = re.sub("'re", ' are', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text
_input1['new_text'] = _input1['text'].apply(lambda x: clean_text(x))
_input0['new_text'] = _input1['text'].apply(lambda x: clean_text(x))
_input1['new_text'][1270]
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
w = WordCloud().generate(_input1['new_text'][1])
plt.imshow(w, interpolation='bilinear')
plt.axis('off')
disaster_train = _input1[_input1['target'] == 1]
nondisaster = _input1[_input1['target'] == 0]
(fig, ax) = plt.subplots(1, 2, figsize=[30, 10])
w1 = WordCloud(background_color='white').generate(''.join(disaster_train['new_text']))
ax[0].imshow(w1)
ax[0].axis('off')
ax[0].set_title('Disaster Tweets', fontsize=40)
w2 = WordCloud(background_color='white').generate(''.join(nondisaster['new_text']))
ax[1].imshow(w2)
ax[1].axis('off')
ax[1].set_title('Non_Disaster Tweets', fontsize=40)
tokens = word_tokenize(_input1['text'][0])
print('normal tokens', tokens)
tokens = [t for t in tokens if t not in stopwords.words('english')]
print('no stipwords', tokens)
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
snowball = nltk.SnowballStemmer(language='english')
print()
porter = [porter.stem(t) for t in tokens]
print('porter', porter)
print()
lan = [lancaster.stem(t) for t in tokens]
print('lancaster', lan)
print()
snowball = [snowball.stem(t) for t in tokens]
print('snowball', snowball)
print()

def clean_text2(text):
    text = re.sub('<.*?>+', '', text)
    token = word_tokenize(text)
    lemm = [lemmatizer.lemmatize(t) for t in token]
    snowball = nltk.SnowballStemmer(language='english')
    snb = [snowball.stem(t) for t in lemm]
    text = ' '.join(snb)
    text = re.sub('via|wa|ha|tt|ve', '', text)
    return text
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
_input1['new_text2'] = _input1['new_text'].apply(lambda x: clean_text2(x))
_input0['new_text2'] = _input0['new_text'].apply(lambda x: clean_text2(x))
"\nfrom nltk.stem import WordNetLemmatizer \n  \nlemmatizer = WordNetLemmatizer() \ntoken=word_tokenize(train['new_text'][1270])\nlemm=[lemmatizer.lemmatize(t) for t in token]\nprint(lemm)\n"
disaster_train = _input1[_input1['target'] == 1]
nondisaster = _input1[_input1['target'] == 0]
(fig, ax) = plt.subplots(1, 2, figsize=[30, 10])
w1 = WordCloud(background_color='white').generate(''.join(disaster_train['new_text2']))
ax[0].imshow(w1)
ax[0].axis('off')
ax[0].set_title('Disaster Tweets', fontsize=40)
w2 = WordCloud(background_color='white').generate(''.join(nondisaster['new_text2']))
ax[1].imshow(w2)
ax[1].axis('off')
ax[1].set_title('Non_Disaster Tweets', fontsize=40)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
cv = CountVectorizer(token_pattern='\\w{2,}', ngram_range=(1, 1), analyzer='word')