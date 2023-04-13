import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import plotly
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from nltk.tokenize import RegexpTokenizer
import nltk
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from imblearn.combine import SMOTETomek
from wordcloud import WordCloud
import plotly.graph_objects as go
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english')
default_tokenizer = RegexpTokenizer('\\w+')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1.shape
_input1.columns
_input1.dtypes
df_target = _input1['target'].value_counts()
fig = go.Figure([go.Pie(labels=df_target.index, values=df_target.values, hole=0.5)])
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=15)
fig.update_layout(title='Disaster Tweets target columns distribution', title_x=0.5)
fig.show()
df_columns = _input1.columns.tolist()
null_value_counts = _input1[df_columns].isnull().sum()
fig = go.Figure(go.Bar(x=null_value_counts.index, y=null_value_counts.values, text=null_value_counts.values, textposition='outside'))
fig.update_layout(title_text='Null value counts', xaxis_title='Column name', yaxis_title='Counts of null values')
fig.show()
_input1['keyword'] = _input1['keyword'].astype(str)
_input1['text'] = _input1[['keyword', 'text']].apply(lambda x: ' '.join(x), axis=1)
_input1 = _input1.drop(['location', 'keyword'], axis=1)

def clean_text(text):
    if text is not None:
        text = re.sub('[0-9]+', '', text)
        text = text.lower()
        text = re.sub('re:', '', text)
        text = re.sub('-', '', text)
        text = re.sub('_', '', text)
        text = re.sub('^https?:\\/\\/.*[\\r\\n]*', '', text, flags=re.MULTILINE)
        text = re.sub('\\S*@\\S*\\s?', '', text, flags=re.MULTILINE)
        text = re.sub('\\[[^]]*\\]', '', text)
        text = re.sub('[^\\w\\s]', '', text)
        text = re.sub('\\n', ' ', text)
        text = re.sub('[0-9]+', '', text)
        p = re.compile('<.*?>')
        text = re.sub("\\'ve", ' have ', text)
        text = re.sub("can't", 'cannot ', text)
        text = re.sub("n't", ' not ', text)
        text = re.sub("I'm", 'I am', text)
        text = re.sub(' m ', ' am ', text)
        text = re.sub("\\'re", ' are ', text)
        text = re.sub("\\'d", ' would ', text)
        text = re.sub("\\'ll", ' will ', text)
        text = p.sub('', text)

    def tokenize_text(text, tokenizer=default_tokenizer):
        token = default_tokenizer.tokenize(text)
        return token

    def remove_stopwords(text, stop_words=default_stopwords):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)

    def stem_text(text, stemmer=default_stemmer):
        tokens = tokenize_text(text)
        return ' '.join([stemmer.stem(t) for t in tokens])
    text = stem_text(text)
    text = remove_stopwords(text)
    return text
_input1['text'] = _input1['text'].apply(clean_text)
tweet_text_list = _input1.text.tolist()
tweet_text_string = ''.join(tweet_text_list)
high_freq_word = pd.Series(' '.join(_input1['text']).split()).value_counts()[:20]
fig = go.Figure(go.Bar(y=high_freq_word.index, x=high_freq_word.values, orientation='h', marker={'color': high_freq_word.values, 'colorscale': 'Viridis'}))
fig.update_layout(title_text='Search most frequent word use in text column', xaxis_title='Count', yaxis_title='Words')
fig.show()
wordcloud_ip = WordCloud(background_color='black', margin=3, width=1800, height=1400, max_words=200).generate(tweet_text_string)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud_ip)
cv = TfidfVectorizer(max_features=1000)
x = cv.fit_transform(_input1['text'])
df1 = pd.DataFrame(x.toarray(), columns=cv.get_feature_names())
_input1 = _input1.drop(['text'], axis=1, inplace=False)
main_df = pd.concat([_input1, df1], axis=1)
main_df.head()
Y = main_df.iloc[:, 1]
X = main_df.iloc[:, 2:]
rfc = RandomForestClassifier(n_jobs=3, oob_score=True, n_estimators=2000, criterion='entropy')