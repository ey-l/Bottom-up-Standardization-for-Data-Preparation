import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
plt.style.use('seaborn')
plt.rcParams['lines.linewidth'] = 1
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score
NBR_STAR = 70
X = pd.read_csv('data/input/nlp-getting-started/train.csv')
y = X['target']
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
X.head()
X['keyword'].fillna('', inplace=True)
X['keyword'] = X['keyword'].map(lambda x: x.replace('%20', ' '))
test['keyword'].fillna('', inplace=True)
test['keyword'] = test['keyword'].map(lambda x: x.replace('%20', ' '))

def clean_text(text):
    """Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers."""
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text
stop_word = list(ENGLISH_STOP_WORDS)
stop_word.append('http')
stop_word.append('https')
stop_word.append('û_')
X['target'].value_counts().plot(kind='barh')


def plot_sample_length_distribution(sample_texts):
    plt.figure(figsize=(10, 10))
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')

plot_sample_length_distribution(X['text'])
keyword_stats = X.groupby('keyword').agg({'text': np.size, 'target': np.mean}).rename(columns={'text': 'Count', 'target': 'Disaster Probability'})
keywords_disaster = keyword_stats.loc[keyword_stats['Disaster Probability'] == 1]
keywords_no_disaster = keyword_stats.loc[keyword_stats['Disaster Probability'] == 0]
keyword_stats.sort_values('Disaster Probability', ascending=False).head(10)
from wordcloud import WordCloud, STOPWORDS
STOPWORDS.add('http')
STOPWORDS.add('https')
STOPWORDS.add('CO')
STOPWORDS.add('û_')
no_disaster_text = ' '.join(X[X['target'] == 0].text.to_numpy().tolist())
real_disaster_text = ' '.join(X[X['target'] == 1].text.to_numpy().tolist())
no_disaster_cloud = WordCloud(stopwords=stop_word, background_color='white').generate(no_disaster_text)
real_disaster_cloud = WordCloud(stopwords=stop_word, background_color='white').generate(real_disaster_text)

def show_word_cloud(cloud, title):
    plt.figure(figsize=(16, 10))
    plt.imshow(cloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')

show_word_cloud(no_disaster_cloud, 'No disaster common words')
show_word_cloud(real_disaster_cloud, 'Real disaster common words')
vect = CountVectorizer(min_df=2, ngram_range=(1, 2), stop_words=stop_word)
X_train = vect.fit_transform(X['text'])
print('Vocabulary size: {}'.format(len(vect.vocabulary_)))
all_ngrams = list(vect.get_feature_names())
num_ngrams = 50
all_counts = X_train.sum(axis=0).tolist()[0]
(all_counts, all_ngrams) = zip(*[(c, n) for (c, n) in sorted(zip(all_counts, all_ngrams), reverse=True)])
ngrams = list(all_ngrams)[:num_ngrams]
counts = list(all_counts)[:num_ngrams]
idx = np.arange(num_ngrams)
plt.figure(figsize=(10, 10))
plt.barh(idx, counts, color='orange')
plt.ylabel('N-grams')
plt.xlabel('Frequencies')
plt.title('Frequency distribution of n-grams')
plt.yticks(idx, ngrams)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=123)
scores = cross_val_score(LogisticRegression(), X_train, y, scoring='f1', cv=cv)
print('*' * NBR_STAR + '\n LogisticRegression on bag of word - cross-validation f1_score: {:.5f}\n'.format(np.mean(scores)) + '*' * NBR_STAR)
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=10, stop_words='english', lowercase=True, use_idf=True, norm=u'l2', smooth_idf=True, ngram_range=(1, 3))
X_train = vect.fit_transform(X['text'])
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
feature_names = np.array(vect.get_feature_names())
print('Features with lowest tfidf:\n{}'.format(feature_names[sorted_by_tfidf[:20]]))
print('Features with highest tfidf: \n{}'.format(feature_names[sorted_by_tfidf[-20:]]))
scores = cross_val_score(LogisticRegression(), X_train, y, scoring='f1', cv=cv)
print('*' * NBR_STAR + '\n LogisticRegression with tfidf - cross-validation f1_score: {:.5f}\n'.format(np.mean(scores)) + '*' * NBR_STAR)
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap

def visualize_coefficients(coefficients, feature_names, n_top_features=25):
    coefficients = coefficients.squeeze()
    if coefficients.ndim > 1:
        raise ValueError('coeffients must be 1d array or column vector, got shape {}'.format(coefficients.shape))
    coefficients = coefficients.ravel()
    if len(coefficients) != len(feature_names):
        raise ValueError("Number of coefficients {} doesn't match number offeature names {}.".format(len(coefficients), len(feature_names)))
    coef = coefficients.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])
    plt.figure(figsize=(20, 7))
    cm = ListedColormap(['#0000aa', '#ff2020'])
    colors = [cm(1) if c < 0 else cm(0) for c in coef[interesting_coefficients]]
    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(np.arange(1, 1 + 2 * n_top_features), feature_names[interesting_coefficients], rotation=60, ha='right')
    plt.ylabel('Coefficient magnitude')
    plt.xlabel('Words')
logreg = LogisticRegression()