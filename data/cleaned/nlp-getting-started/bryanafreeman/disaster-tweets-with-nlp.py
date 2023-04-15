import re
import string
import urllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.tokenize import TweetTokenizer, sent_tokenize, word_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
warnings.filterwarnings('ignore')
pd.options.display.max_rows = 10
pd.options.display.max_columns = 20
print('numpy version: {}'.format(np.__version__))
print('pandas version: {}'.format(pd.__version__))
print('seaborn version: {}\n'.format(sns.__version__))
sns.set_style('whitegrid')
flatui = ['#9b59b6', '#3498db', '#95a5a6', '#e74c3c', '#34495e', '#2ecc71']
sns.set_palette(flatui)
sns.palplot(sns.color_palette())
df_train = pd.read_csv('data/input/nlp-getting-started/train.csv', encoding='utf8')
print('Train data loaded.')
clean_copy = df_train.copy()
df_test = pd.read_csv('data/input/nlp-getting-started/test.csv', encoding='utf8')
print('Test data loaded.')
sample_sub = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv', encoding='utf8')
print('There are {} of records in the train data set.'.format(len(df_train.index)))
print('There are {} of records in the test data set.'.format(len(df_test.index)))
df_train.head()
df_train.info()
df_train.isnull().sum()
target_value_counts = df_train.target.value_counts()
print(target_value_counts)
sns.set_style('ticks')
(fig, ax) = plt.subplots()
fig.set_size_inches(11, 8)
clrs = ['#2ecc71', '#e74c3c']
sns.barplot(x=target_value_counts.index, y=target_value_counts, capsize=0.3, palette=clrs)
plt.xlabel('0=Fake News    or    1=Real Disaster')
plt.ylabel('Number of Tweets')
plt.title('Distribution of Test Data')

cols_with_missing = ['keyword', 'location']
train_empties = df_train[cols_with_missing].isnull().sum() / len(df_train) * 100
(fig, ax) = plt.subplots()
fig.set_size_inches(11, 8)
clrs = ['#3498db', '#e74c3c']
sns.barplot(x=train_empties.index, y=train_empties.values, ax=ax, capsize=0.3, palette=clrs)
ax.set_ylabel('Percent Missing Values', labelpad=20)
ax.set_yticks(np.arange(0, 40, 5))
ax.set_ylim((0, 35))
ax.set_title('Missing Keywords and Locations', fontsize=13)

keyword_value_counts = df_train['keyword'].value_counts()
print('There are {} unique keywords.\n'.format(len(keyword_value_counts)))
print(keyword_value_counts)
top_25_kw = keyword_value_counts[:25]
tick_range = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
(fig, ax) = plt.subplots()
fig.set_size_inches(16, 8)
sns.barplot(y=top_25_kw.values, x=top_25_kw.index, palette=flatui)
plt.xlabel('Keyword')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.yticks(ticks=tick_range, rotation=0)
plt.title('Frequency of Keyword Use - Top 25')

true_ratios = df_train.groupby('keyword')['target'].mean().sort_values(ascending=False)
(fig, ax) = plt.subplots()
fig.set_size_inches(16, 8)
sns.barplot(x=true_ratios.index[:25], y=true_ratios.values[:25], ax=ax, palette=flatui)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.xlabel('Keyword')
plt.ylabel('True-False Ratio')
plt.title('Top 25 Keywords for Fake Disasters')

loc_value_counts = df_train['location'].value_counts()
print('There are {} unique keywords.\n'.format(len(loc_value_counts)))
print(loc_value_counts)
top_25_loc = loc_value_counts[:25]
tick_range = [0, 20, 40, 60, 80, 100, 120]
(fig, ax) = plt.subplots()
fig.set_size_inches(16, 8)
sns.barplot(y=top_25_loc.values, x=top_25_loc.index, palette=flatui)
plt.xlabel('Location')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.yticks(ticks=tick_range, rotation=0)
plt.title('Frequency of Locations - Top 25')


def wc(x, stop_words, max_words, bgcolor, plot_title):
    plt.figure(figsize=(16, 8))
    wc = WordCloud(background_color=bgcolor, stopwords=stop_words, max_words=max_words, max_font_size=50).generate(str(x))
    wc.generate(' '.join(x))
    plt.title(plot_title)
    plt.imshow(wc)
    plt.axis('off')
max_words = 500
stop_words = ['https', 'co', 'RT', 'http', 'hi', 'amp', 'ha'] + list(STOPWORDS)
wc(df_train[df_train['target'] == 1]['text'], stop_words, max_words, 'black', 'Most Frequent Words - Real')
wc(df_train[df_train['target'] == 0]['text'], stop_words, max_words, 'black', 'Most Frequent Words - Fake')
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(16, 8))
tweet_len_real = df_train[df_train['target'] == 1]['text'].str.len()
sns.distplot(tweet_len_real, ax=ax1, color='#e74c3c')
ax1.set_title('Real Disaster')
tweet_len_fake = df_train[df_train['target'] == 0]['text'].str.len()
sns.distplot(tweet_len_fake, ax=ax2, color='#2ecc71')
ax2.set_title('Fake Disaster')
fig.suptitle('Length in Characters')

(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(16, 8))
words_real = df_train[df_train['target'] == 1]['text'].str.split().apply(lambda x: [len(i) for i in x])
words_fake = df_train[df_train['target'] == 0]['text'].str.split().apply(lambda x: [len(i) for i in x])
sns.distplot(words_real.map(lambda x: np.mean(x)), ax=ax1, color='#e74c3c')
ax1.set_title('Real Disaster')
sns.distplot(words_fake.map(lambda x: np.mean(x)), ax=ax2, color='#2ecc71')
ax2.set_title('Fake Disaster')
fig.suptitle('Average Length of Tweets in Words')

def avg_word_len(text):
    words = word_tokenize(text)
    word_lens = [len(w) for w in words]
    return round(np.mean(word_lens), 1)

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [w for w in text if not w in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

def text_feature_eng(x):
    x['clean_text'] = x['text'].apply(lambda x: clean_text(x))
    tweek_tzr = TweetTokenizer()
    x['word_cnt'] = x['clean_text'].apply(lambda t: len(tweek_tzr.tokenize(t.lower())))
    x['char_cnt'] = x['clean_text'].apply(lambda c: len(c))
    hashtag_re = '#\\w+'
    x['hashtag_ct'] = x['text'].apply(lambda h: len(regexp_tokenize(h, hashtag_re)))
    x['avg_word_len'] = x['clean_text'].apply(avg_word_len)
    num_re = '(\\d+\\.?,?\\s?\\d+)'
    x['num_cnt'] = x['text'].apply(lambda n: len(regexp_tokenize(n, num_re)))
    punct_re = '[^\\w\\s]'
    x['punct_cnt'] = x['text'].apply(lambda p: len(regexp_tokenize(p, punct_re)))
    mention_re = '@\\w+'
    x['mention_cnt'] = x['text'].apply(lambda m: len(regexp_tokenize(m, mention_re)))
    x['bow'] = x['clean_text'].apply(lambda t: [w for w in tweek_tzr.tokenize(t.lower())])
    x['words_only'] = x['bow'].apply(lambda w: [t for t in w if t.isalpha()])
    x['stopwords'] = x['bow'].apply(lambda x: [t for t in x if t in stopwords.words('english')])
    x['emojis'] = x['text'].apply(lambda comment: sum((comment.count(e) for e in (':-)', ':)', ';-)', ';)', ':(', ':-('))))
    x['no_keywords'] = x['keyword'].isna().astype(int)
    x['no_location'] = x['location'].isna().astype(int)
    x.drop('text', axis=1, inplace=True)
    x.avg_word_len.fillna(0)
    return x
df_train = clean_copy.copy()
X = df_train.copy()
X_proc = text_feature_eng(X)
print('X_proc shape: {}'.format(X_proc.shape))
X_proc.head()
all_features = [col for col in X_proc.columns.values if col not in ['id', 'target']]
num_features = [col for col in X_proc.columns.values if col not in ['id', 'target', 'bow', 'words_only', 'stopwords', 'clean_text', 'keyword', 'location', 'no_keywords', 'no_location']]
SEED = 37
SPLIT = 0.8
(X_train, X_val, y_train, y_val) = train_test_split(X_proc[all_features], X_proc['target'], train_size=SPLIT, shuffle=True, random_state=SEED)
print('{} training records'.format(len(X_train)))
print('{} training labels'.format(len(y_train)))
print('{} validation records'.format(len(X_val)))
print('{} validation labels'.format(len(y_val)))

class TextSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]

class NumericSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]
tfidf_pipeline = Pipeline([('selector', TextSelector(key='clean_text')), ('tfidf', TfidfVectorizer())])
length_pipeline = Pipeline([('selector', NumericSelector(key='avg_word_len')), ('standard', StandardScaler())])
words_pipeline = Pipeline([('selector', NumericSelector(key='word_cnt')), ('standard', StandardScaler())])
char_pipeline = Pipeline([('selector', NumericSelector(key='char_cnt')), ('standard', StandardScaler())])
num_pipeline = Pipeline([('selector', NumericSelector(key='num_cnt')), ('standard', StandardScaler())])
punct_pipeline = Pipeline([('selector', NumericSelector(key='punct_cnt')), ('standard', StandardScaler())])
feature_pipeline = FeatureUnion([('tfidf', tfidf_pipeline), ('length', length_pipeline), ('words', words_pipeline), ('chars', char_pipeline), ('nums', num_pipeline), ('punct', punct_pipeline)])
feature_processing = Pipeline([('features', feature_pipeline)])
feature_processing.fit_transform(X_train)
print('X_train shape: {}'.format(X_train.shape))
tuned_parameters = {'kernel': ['linear'], 'C': [1, 5, 10], 'cache_size': [100, 200, 400], 'degree': [2, 5, 10]}
scores = ['precision', 'recall']
for score in scores:
    print('Tuning hyper-parameters for %s \n' % score)
    print('Creating pipeline instance.')
    sentiment_pipeline = Pipeline([('features', feature_pipeline), ('classifier', GridSearchCV(SVC(), tuned_parameters, scoring='%s_macro' % score, verbose=10, n_jobs=-1, cv=3))])
    print('Fitting the model.')