import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk.tokenize import regexp_tokenize, sent_tokenize, word_tokenize, TweetTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
train.head()
train.shape
train.info()
train.target.value_counts()
train.target.value_counts() / len(train)
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=train)
plt.title('No.of Disaster Vs Non-Disaster Tweets')
plt.xlabel('Target', fontsize=11)

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)
tweet_words = ''
for val in train.text:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    tweet_words += ' '.join(tokens) + ' '
plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stopwords, min_font_size=10).generate(tweet_words)
plt.imshow(wordcloud)
plt.axis('off')


def remove_urls(text):
    url_pattern = re.compile('https?://\\S+|www\\.\\S+')
    return url_pattern.sub('', text)

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub('', text)
EMOTICONS = {u':‑\\)': 'Happy face or smiley', u':\\)': 'Happy face or smiley', u':-\\]': 'Happy face or smiley', u':\\]': 'Happy face or smiley', u':-3': 'Happy face smiley', u':3': 'Happy face smiley', u':->': 'Happy face smiley', u':>': 'Happy face smiley', u'8-\\)': 'Happy face smiley', u':o\\)': 'Happy face smiley', u':-\\}': 'Happy face smiley', u':\\}': 'Happy face smiley', u':-\\)': 'Happy face smiley', u':c\\)': 'Happy face smiley', u':\\^\\)': 'Happy face smiley', u'=\\]': 'Happy face smiley', u'=\\)': 'Happy face smiley', u':‑D': 'Laughing, big grin or laugh with glasses', u':D': 'Laughing, big grin or laugh with glasses', u'8‑D': 'Laughing, big grin or laugh with glasses', u'8D': 'Laughing, big grin or laugh with glasses', u'X‑D': 'Laughing, big grin or laugh with glasses', u'XD': 'Laughing, big grin or laugh with glasses', u'=D': 'Laughing, big grin or laugh with glasses', u'=3': 'Laughing, big grin or laugh with glasses', u'B\\^D': 'Laughing, big grin or laugh with glasses', u':-\\)\\)': 'Very happy', u':‑\\(': 'Frown, sad, andry or pouting', u':-\\(': 'Frown, sad, andry or pouting', u':\\(': 'Frown, sad, andry or pouting', u':‑c': 'Frown, sad, andry or pouting', u':c': 'Frown, sad, andry or pouting', u':‑<': 'Frown, sad, andry or pouting', u':<': 'Frown, sad, andry or pouting', u':‑\\[': 'Frown, sad, andry or pouting', u':\\[': 'Frown, sad, andry or pouting', u':-\\|\\|': 'Frown, sad, andry or pouting', u'>:\\[': 'Frown, sad, andry or pouting', u':\\{': 'Frown, sad, andry or pouting', u':@': 'Frown, sad, andry or pouting', u'>:\\(': 'Frown, sad, andry or pouting', u":'‑\\(": 'Crying', u":'\\(": 'Crying', u":'‑\\)": 'Tears of happiness', u":'\\)": 'Tears of happiness', u"D‑':": 'Horror', u'D:<': 'Disgust', u'D:': 'Sadness', u'D8': 'Great dismay', u'D;': 'Great dismay', u'D=': 'Great dismay', u'DX': 'Great dismay', u':‑O': 'Surprise', u':O': 'Surprise', u':‑o': 'Surprise', u':o': 'Surprise', u':-0': 'Shock', u'8‑0': 'Yawn', u'>:O': 'Yawn', u':-\\*': 'Kiss', u':\\*': 'Kiss', u':X': 'Kiss', u';‑\\)': 'Wink or smirk', u';\\)': 'Wink or smirk', u'\\*-\\)': 'Wink or smirk', u'\\*\\)': 'Wink or smirk', u';‑\\]': 'Wink or smirk', u';\\]': 'Wink or smirk', u';\\^\\)': 'Wink or smirk', u':‑,': 'Wink or smirk', u';D': 'Wink or smirk', u':‑P': 'Tongue sticking out, cheeky, playful or blowing a raspberry', u':P': 'Tongue sticking out, cheeky, playful or blowing a raspberry', u'X‑P': 'Tongue sticking out, cheeky, playful or blowing a raspberry', u'XP': 'Tongue sticking out, cheeky, playful or blowing a raspberry', u':‑Þ': 'Tongue sticking out, cheeky, playful or blowing a raspberry', u':Þ': 'Tongue sticking out, cheeky, playful or blowing a raspberry', u':b': 'Tongue sticking out, cheeky, playful or blowing a raspberry', u'd:': 'Tongue sticking out, cheeky, playful or blowing a raspberry', u'=p': 'Tongue sticking out, cheeky, playful or blowing a raspberry', u'>:P': 'Tongue sticking out, cheeky, playful or blowing a raspberry', u':‑/': 'Skeptical, annoyed, undecided, uneasy or hesitant', u':/': 'Skeptical, annoyed, undecided, uneasy or hesitant', u':-[.]': 'Skeptical, annoyed, undecided, uneasy or hesitant', u'>:[(\\\\)]': 'Skeptical, annoyed, undecided, uneasy or hesitant', u'>:/': 'Skeptical, annoyed, undecided, uneasy or hesitant', u':[(\\\\)]': 'Skeptical, annoyed, undecided, uneasy or hesitant', u'=/': 'Skeptical, annoyed, undecided, uneasy or hesitant', u'=[(\\\\)]': 'Skeptical, annoyed, undecided, uneasy or hesitant', u':L': 'Skeptical, annoyed, undecided, uneasy or hesitant', u'=L': 'Skeptical, annoyed, undecided, uneasy or hesitant', u':S': 'Skeptical, annoyed, undecided, uneasy or hesitant', u':‑\\|': 'Straight face', u':\\|': 'Straight face', u':$': 'Embarrassed or blushing', u':‑x': 'Sealed lips or wearing braces or tongue-tied', u':x': 'Sealed lips or wearing braces or tongue-tied', u':‑#': 'Sealed lips or wearing braces or tongue-tied', u':#': 'Sealed lips or wearing braces or tongue-tied', u':‑&': 'Sealed lips or wearing braces or tongue-tied', u':&': 'Sealed lips or wearing braces or tongue-tied', u'O:‑\\)': 'Angel, saint or innocent', u'O:\\)': 'Angel, saint or innocent', u'0:‑3': 'Angel, saint or innocent', u'0:3': 'Angel, saint or innocent', u'0:‑\\)': 'Angel, saint or innocent', u'0:\\)': 'Angel, saint or innocent', u':‑b': 'Tongue sticking out, cheeky, playful or blowing a raspberry', u'0;\\^\\)': 'Angel, saint or innocent', u'>:‑\\)': 'Evil or devilish', u'>:\\)': 'Evil or devilish', u'\\}:‑\\)': 'Evil or devilish', u'\\}:\\)': 'Evil or devilish', u'3:‑\\)': 'Evil or devilish', u'3:\\)': 'Evil or devilish', u'>;\\)': 'Evil or devilish', u'\\|;‑\\)': 'Cool', u'\\|‑O': 'Bored', u':‑J': 'Tongue-in-cheek', u'#‑\\)': 'Party all night', u'%‑\\)': 'Drunk or confused', u'%\\)': 'Drunk or confused', u':-###..': 'Being sick', u':###..': 'Being sick', u'<:‑\\|': 'Dump', u'\\(>_<\\)': 'Troubled', u'\\(>_<\\)>': 'Troubled', u"\\(';'\\)": 'Baby', u'\\(\\^\\^>``': 'Nervous or Embarrassed or Troubled or Shy or Sweat drop', u'\\(\\^_\\^;\\)': 'Nervous or Embarrassed or Troubled or Shy or Sweat drop', u'\\(-_-;\\)': 'Nervous or Embarrassed or Troubled or Shy or Sweat drop', u'\\(~_~;\\) \\(・\\.・;\\)': 'Nervous or Embarrassed or Troubled or Shy or Sweat drop', u'\\(-_-\\)zzz': 'Sleeping', u'\\(\\^_-\\)': 'Wink', u'\\(\\(\\+_\\+\\)\\)': 'Confused', u'\\(\\+o\\+\\)': 'Confused', u'\\(o\\|o\\)': 'Ultraman', u'\\^_\\^': 'Joyful', u'\\(\\^_\\^\\)/': 'Joyful', u'\\(\\^O\\^\\)／': 'Joyful', u'\\(\\^o\\^\\)／': 'Joyful', u'\\(__\\)': 'Kowtow as a sign of respect, or dogeza for apology', u'_\\(\\._\\.\\)_': 'Kowtow as a sign of respect, or dogeza for apology', u'<\\(_ _\\)>': 'Kowtow as a sign of respect, or dogeza for apology', u'<m\\(__\\)m>': 'Kowtow as a sign of respect, or dogeza for apology', u'm\\(__\\)m': 'Kowtow as a sign of respect, or dogeza for apology', u'm\\(_ _\\)m': 'Kowtow as a sign of respect, or dogeza for apology', u"\\('_'\\)": 'Sad or Crying', u'\\(/_;\\)': 'Sad or Crying', u'\\(T_T\\) \\(;_;\\)': 'Sad or Crying', u'\\(;_;': 'Sad of Crying', u'\\(;_:\\)': 'Sad or Crying', u'\\(;O;\\)': 'Sad or Crying', u'\\(:_;\\)': 'Sad or Crying', u'\\(ToT\\)': 'Sad or Crying', u';_;': 'Sad or Crying', u';-;': 'Sad or Crying', u';n;': 'Sad or Crying', u';;': 'Sad or Crying', u'Q\\.Q': 'Sad or Crying', u'T\\.T': 'Sad or Crying', u'QQ': 'Sad or Crying', u'Q_Q': 'Sad or Crying', u'\\(-\\.-\\)': 'Shame', u'\\(-_-\\)': 'Shame', u'\\(一一\\)': 'Shame', u'\\(；一_一\\)': 'Shame', u'\\(=_=\\)': 'Tired', u'\\(=\\^\\·\\^=\\)': 'cat', u'\\(=\\^\\·\\·\\^=\\)': 'cat', u'=_\\^=\t': 'cat', u'\\(\\.\\.\\)': 'Looking down', u'\\(\\._\\.\\)': 'Looking down', u'\\^m\\^': 'Giggling with hand covering mouth', u'\\(\\・\\・?': 'Confusion', u'\\(?_?\\)': 'Confusion', u'>\\^_\\^<': 'Normal Laugh', u'<\\^!\\^>': 'Normal Laugh', u'\\^/\\^': 'Normal Laugh', u'\\（\\*\\^_\\^\\*）': 'Normal Laugh', u'\\(\\^<\\^\\) \\(\\^\\.\\^\\)': 'Normal Laugh', u'\\(^\\^\\)': 'Normal Laugh', u'\\(\\^\\.\\^\\)': 'Normal Laugh', u'\\(\\^_\\^\\.\\)': 'Normal Laugh', u'\\(\\^_\\^\\)': 'Normal Laugh', u'\\(\\^\\^\\)': 'Normal Laugh', u'\\(\\^J\\^\\)': 'Normal Laugh', u'\\(\\*\\^\\.\\^\\*\\)': 'Normal Laugh', u'\\(\\^—\\^\\）': 'Normal Laugh', u'\\(#\\^\\.\\^#\\)': 'Normal Laugh', u'\\（\\^—\\^\\）': 'Waving', u'\\(;_;\\)/~~~': 'Waving', u'\\(\\^\\.\\^\\)/~~~': 'Waving', u'\\(-_-\\)/~~~ \\($\\·\\·\\)/~~~': 'Waving', u'\\(T_T\\)/~~~': 'Waving', u'\\(ToT\\)/~~~': 'Waving', u'\\(\\*\\^0\\^\\*\\)': 'Excited', u'\\(\\*_\\*\\)': 'Amazed', u'\\(\\*_\\*;': 'Amazed', u'\\(\\+_\\+\\) \\(@_@\\)': 'Amazed', u'\\(\\*\\^\\^\\)v': 'Laughing,Cheerful', u'\\(\\^_\\^\\)v': 'Laughing,Cheerful', u'\\(\\(d[-_-]b\\)\\)': 'Headphones,Listening to music', u'\\(-"-\\)': 'Worried', u'\\(ーー;\\)': 'Worried', u'\\(\\^0_0\\^\\)': 'Eyeglasses', u'\\(\\＾ｖ\\＾\\)': 'Happy', u'\\(\\＾ｕ\\＾\\)': 'Happy', u'\\(\\^\\)o\\(\\^\\)': 'Happy', u'\\(\\^O\\^\\)': 'Happy', u'\\(\\^o\\^\\)': 'Happy', u'\\)\\^o\\^\\(': 'Happy', u':O o_O': 'Surprised', u'o_0': 'Surprised', u'o\\.O': 'Surpised', u'\\(o\\.o\\)': 'Surprised', u'oO': 'Surprised', u'\\(\\*￣m￣\\)': 'Dissatisfied', u'\\(‘A`\\)': 'Snubbed or Deflated'}

def remove_emoji(string):
    emoji_pattern = re.compile('[😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿✂-➰Ⓜ-🉑]+', flags=re.UNICODE)
    return emoji_pattern.sub('', string)

def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join((k for k in EMOTICONS)) + u')')
    return emoticon_pattern.sub('', text)
PUNCT_TO_REMOVE = string.punctuation

def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
train['text_clean'] = train['text'].apply(lambda x: remove_html(x))
train['text_clean'] = train['text'].apply(lambda x: remove_urls(x))
train['text_clean'] = train['text'].apply(lambda x: remove_emoticons(x))
train['text_clean'] = train['text'].apply(lambda x: remove_emoji(x))
train['text_clean'] = train['text'].apply(lambda x: remove_punctuation(x))

def find_hashtags(tweet):
    return ' '.join([match.group(0)[1:] for match in re.finditer('#\\w+', tweet)]) or 'no'

def process_text(df):
    df['hashtags'] = df['text'].apply(lambda x: find_hashtags(x))
    return df
train = process_text(train)
tweet_words = ''
for val in train.text_clean:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    tweet_words += ' '.join(tokens) + ' '
plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stopwords, min_font_size=10).generate(tweet_words)
plt.imshow(wordcloud)
plt.axis('off')

disastertweet_words = ''
for val in train[train['target'] == 1].text_clean:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    disastertweet_words += ' '.join(tokens) + ' '
plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stopwords, min_font_size=10).generate(disastertweet_words)
plt.imshow(wordcloud)
plt.title('Word Cloud of tweets if real disaster')
plt.axis('off')

nodisastertweet_words = ''
for val in train[train['target'] == 0].text_clean:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    nodisastertweet_words += ' '.join(tokens) + ' '
plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stopwords, min_font_size=10).generate(nodisastertweet_words)
plt.imshow(wordcloud)
plt.title('Word Cloud of tweets if no disaster')
plt.axis('off')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
tfidf_vectorizer_text_clean = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), min_df=5)
count_vectorizer_hashtags = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), min_df=5)
tfidf_text_clean = tfidf_vectorizer_text_clean.fit_transform(train.text_clean)
count_hashtags = count_vectorizer_hashtags.fit_transform(train.hashtags)
train_text_clean = pd.DataFrame(tfidf_text_clean.toarray(), columns=tfidf_vectorizer_text_clean.get_feature_names())
train_hashtags = pd.DataFrame(count_hashtags.toarray(), columns=count_vectorizer_hashtags.get_feature_names())
print(train_text_clean.shape, train_hashtags.shape)
train = train.join(train_text_clean, rsuffix='_count_text_clean')
train = train.join(train_hashtags, rsuffix='_count_hashtags')
print(train.shape)
features_to_drop = ['id', 'keyword', 'location', 'text', 'text_clean', 'hashtags', 'target_count_text_clean']
final_df = train.drop(columns=features_to_drop, axis=1)
final_df.shape
y = final_df['target']
X = final_df.drop('target', axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=99, stratify=y)
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
logreg = LogisticRegression(solver='liblinear', penalty='l2')