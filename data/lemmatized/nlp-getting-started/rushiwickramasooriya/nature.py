import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium import plugins
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import missingno as msno
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print('libraries successfully imported')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
print('Data successfully imported')
data_set = pd.concat([_input1, _input0], ignore_index=True)
print('Data combined successfully')
data_set.head()
data_set.shape
data_set.info()
data_set.describe()
fig = px.bar(data_set, x=data_set.target.value_counts().index, y=data_set.target.value_counts())
fig.update_layout(xaxis_title='Target', yaxis_title='Count', title={'text': 'Distribution Target Coloumn', 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig.show()
data = data_set['target'].value_counts()
labels = data_set['target'].value_counts().keys()
fig = plt.figure(figsize=(10, 7))
plt.pie(data, labels=labels, autopct='%1.1f%%', colors=sns.color_palette('Set3'))
disaster_keywords = data_set[data_set['target'] == 1]['keyword'].value_counts()
disaster_keywords = disaster_keywords[0:30]
fig = px.scatter(disaster_keywords, x=disaster_keywords.values, y=disaster_keywords.index, size=disaster_keywords.values)
fig.update_layout(xaxis_title='counts', yaxis_title='keywords', title={'text': 'Distribution in Disaster Tweets for Top 30 Keywords', 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig.show()
non_disaster_keywords = data_set.loc[data_set['target'] == 0]['keyword'].value_counts()
plt.figure(figsize=(15, 6))
sns.barplot(y=non_disaster_keywords[0:20], x=non_disaster_keywords[0:20].index, palette='Pastel1')
plt.title('Distribution in Non-Disaster Tweets for Top 20 Keywords')
plt.xticks(rotation=90)
new_data_set = pd.DataFrame()
new_data_set['location'] = data_set['location'].value_counts()[:10].index
new_data_set['count'] = data_set['location'].value_counts()[:10].values
geolocator = Nominatim(user_agent='a@gmail.com')
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)
lat = {}
long = {}
for i in new_data_set['location']:
    location = geocode(i)
    lat[i] = location.latitude
    long[i] = location.longitude
new_data_set['latitude'] = new_data_set['location'].map(lat)
new_data_set['longitude'] = new_data_set['location'].map(long)
map = folium.Map(location=[10.0, 10.0], tiles='CartoDB dark_matter', zoom_start=1.5)
markers = []
title = '<h1 align = "center" style = "font-size: 15px"><b>Top 10 Tweet Locations</b></h1>'
for (i, r) in new_data_set.iterrows():
    loss = r['count']
    if r['count'] > 0:
        counts = r['count'] * 0.4
        folium.CircleMarker([float(r['latitude']), float(r['longitude'])], radius=float(counts), color='lightcoral', fill=True).add_to(map)
map.get_root().html.add_child(folium.Element(title))
map
new_data_set = pd.DataFrame()
new_data_set['location'] = data_set[data_set['target'] == 0]['location'].value_counts()[:10].index
new_data_set['count'] = data_set[data_set['target'] == 0]['location'].value_counts()[:10].values
geolocator = Nominatim(user_agent='a@gmail.com')
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)
lat = {}
long = {}
for i in new_data_set['location']:
    location = geocode(i)
    lat[i] = location.latitude
    long[i] = location.longitude
new_data_set['latitude'] = new_data_set['location'].map(lat)
new_data_set['longitude'] = new_data_set['location'].map(long)
map = folium.Map(location=[10.0, 10.0], zoom_start=1.5)
markers = []
title = '<h1 align = "center" style = "font-size: 15px"><b>Top 10 Non-Disaster Tweet Locations</b></h1>'
for (i, r) in new_data_set.iterrows():
    loss = r['count']
    if r['count'] > 0:
        counts = r['count'] * 0.4
        folium.CircleMarker([float(r['latitude']), float(r['longitude'])], radius=float(counts), color='darkblue', fill=True).add_to(map)
map.get_root().html.add_child(folium.Element(title))
map
fig = make_subplots(rows=1, cols=1, subplot_titles='Location Distribution in Non-Disaster Tweets')
fig.add_trace(go.Bar(x=data_set[data_set['target'] == 0]['location'].value_counts()[:10].index, y=data_set[data_set['target'] == 0]['location'].value_counts()[:10]))
fig.update_layout(height=500, width=1000, title={'text': 'Location Distribution in Non-Disaster Tweets', 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, showlegend=False)
fig.show()
new_data_set = pd.DataFrame()
new_data_set['location'] = data_set[data_set['target'] == 1]['location'].value_counts()[:10].index
new_data_set['count'] = data_set[data_set['target'] == 1]['location'].value_counts()[:10].values
geolocator = Nominatim(user_agent='a@gmail.com')
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)
lat = {}
long = {}
for i in new_data_set['location']:
    location = geocode(i)
    lat[i] = location.latitude
    long[i] = location.longitude
new_data_set['latitude'] = new_data_set['location'].map(lat)
new_data_set['longitude'] = new_data_set['location'].map(long)
map = folium.Map(location=[10.0, 10.0], zoom_start=1.5)
markers = []
title = '<h1 align = "center" style = "font-size: 15px"><b>Top 10 Disaster Tweet Locations</b></h1>'
for (i, r) in new_data_set.iterrows():
    loss = r['count']
    if r['count'] > 0:
        counts = r['count'] * 0.4
        folium.CircleMarker([float(r['latitude']), float(r['longitude'])], radius=float(counts), color='red', fill=True).add_to(map)
map.get_root().html.add_child(folium.Element(title))
map
fig = make_subplots(rows=1, cols=1, subplot_titles='Location Distribution in Disaster Tweets')
fig.add_trace(go.Bar(x=data_set[data_set['target'] == 1]['location'].value_counts()[:10].index, y=data_set[data_set['target'] == 1]['location'].value_counts()[:10]), row=1, col=1)
fig.update_layout(height=500, width=1000, title={'text': 'Location Distribution in Disaster Tweets', 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, showlegend=False)
fig.show()
data_set['word_count'] = data_set['text'].apply(lambda x: len(str(x).split()))
data_set['unique_word_count'] = data_set['text'].apply(lambda x: len(set(str(x).split())))
data_set['stop_word_count'] = data_set['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
data_set['url_count'] = data_set['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
data_set['mean_word_length'] = data_set['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
data_set['char_count'] = data_set['text'].apply(lambda x: len(str(x)))
data_set['punctuation_count'] = data_set['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
data_set['hashtag_count'] = data_set['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
data_set['mention_count'] = data_set['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
text_features = ['word_count', 'unique_word_count', 'stop_word_count', 'url_count', 'mean_word_length', 'char_count', 'punctuation_count', 'hashtag_count', 'mention_count']
disaster_tweets = data_set['target'] == 1
(fig, axes) = plt.subplots(ncols=2, nrows=len(text_features), figsize=(20, 50), dpi=100)
for (i, feature) in enumerate(text_features):
    sns.histplot(data_set.loc[~disaster_tweets][feature], label='Not Disaster', ax=axes[i][0], color='#50C878')
    sns.histplot(data_set.loc[disaster_tweets][feature], label='Disaster', ax=axes[i][0], color='#FF5733')
    sns.histplot(data_set[feature], label='Data Set', ax=axes[i][1])
    for j in range(2):
        axes[i][j].set_xlabel('')
        axes[i][j].tick_params(axis='x', labelsize=12)
        axes[i][j].tick_params(axis='y', labelsize=12)
        axes[i][j].legend()
    axes[i][0].set_title(f'{feature} target distribution in the data set', fontsize=13)
    axes[i][1].set_title(f'{feature} distribution', fontsize=13)
plt.figure(figsize=(15, 10))
wc = WordCloud(max_words=1000, background_color='white')
wc.generate(' '.join((word for word in data_set.loc[data_set['target'] == 0, 'text'])))
plt.imshow(wc)
plt.axis('off')
plt.title('Non-Disaster Tweets')
plt.figure(figsize=(15, 10))
wc = WordCloud(max_words=1000, background_color='black')
wc.generate(' '.join((word for word in data_set.loc[data_set['target'] == 1, 'text'])))
plt.imshow(wc)
plt.axis('off')
plt.title('Disaster Tweets')
data_set.duplicated().sum()
data_set.isnull().any()
total = data_set.isnull().sum().sort_values(ascending=False)
percentage = (data_set.isnull().sum() / data_set.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percentage], axis=1, keys=['Total', 'Percentages '])
missing_data.head(data_set.shape[1])
data_set['target'].isnull().sum()
data_set['text'].head(20)

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\\d+', '', text)
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    pattern = re.compile('@\\w+')
    text = re.sub(pattern, '', text)
    text = re.sub(' +', ' ', text).strip()
    text = re.sub('[😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿✂-➰Ⓜ-🉑]+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text
data_set['text'] = data_set['text'].apply(clean_text)
data_set['text'].head(20)
tokenizer = nltk.tokenize.RegexpTokenizer('\\w+')
data_set['text'] = data_set['text'].apply(lambda x: tokenizer.tokenize(x))
data_set['text'].head()

def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words
data_set['text'] = data_set['text'].apply(lambda x: remove_stopwords(x))
data_set['text'].head()
lemmatizer = WordNetLemmatizer()

def preprocessData(text):
    text = ' '.join((lemmatizer.lemmatize(word) for word in text))
    return text
data_set['text'] = data_set['text'].apply(preprocessData)
data_set['text'].head(20)
data_set['keyword'] = data_set['keyword'].fillna('missing')
data_set['location'] = data_set['location'].fillna('unknown')
data_set.isnull().sum()
new_data_set = data_set.dropna()
new_data_set
total = new_data_set.isnull().sum().sort_values(ascending=False)
percentage = (new_data_set.isnull().sum() / data_set.isnull().count()).sort_values(ascending=False)
data_set = pd.concat([total, percentage], axis=1, keys=['Total', 'Percentages '])
data_set.head(data_set.shape[1])
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction
count_vectorizer = feature_extraction.text.CountVectorizer()
example_train_vectors = count_vectorizer.fit_transform(new_data_set['text'][0:5])
print(example_train_vectors[0].todense().shape)
print(example_train_vectors[0].todense())
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(new_data_set['text'])
y = new_data_set['target'].values
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=40)
linear_svm = LinearSVC()