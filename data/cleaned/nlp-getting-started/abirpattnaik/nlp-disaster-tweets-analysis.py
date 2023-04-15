import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import regex
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_data.head()
print('The train set contains {0} rows and {1} columns '.format(train_data.shape[0], train_data.shape[1]))
ax = sns.countplot(data=train_data, x=train_data['target'])
plt.xlabel('Target Variable- Disaster or not disaster tweet')
plt.ylabel('Count of tweets')
plt.title('Count of disaster and non-disaster tweets')
total = len(train_data)
for p in ax.patches:
    ax.annotate('{:.1f}%'.format(100 * p.get_height() / total), (p.get_x() + 0.1, p.get_height() + 5))
train_data['keyword'] = train_data.keyword.str.replace('%20', '_')
train_data['keyword'] = train_data['keyword'].replace(np.nan, '', regex=True)
keyword_count = pd.DataFrame(train_data.groupby(['keyword', 'target']).agg(['count']).sort_values(by=('id', 'count'), ascending=False)['id', 'count'])
keyword_count.columns
"\nkeyword_count['keyword']=keyword_count.index\nkeyword_count.columns=keyword_count.columns.droplevel()\nkeyword_count.columns=['count','keyword']\n\nfor value in range(0,len(keyword_count)):\n    #print(keyword_count.keyword[value][0])\n    if 'keyword_value' not in keyword_count.columns:\n        keyword_count['keyword_value']=keyword_count.keyword[0][0]\n    else:\n        keyword_count['keyword_value'][value]=keyword_count.keyword[value][0]\n    #print(keyword_count.keyword[value][1])\n    if 'target_value' not in keyword_count.columns:\n        keyword_count['target_value']=keyword_count.keyword[0][1]\n    else:\n        keyword_count['target_value'][value]=keyword_count.keyword[value][1]\n\nif 'keyword' in keyword_count.columns:\n    keyword_count=keyword_count.drop(['keyword'],axis=1)\n#Index(['count', 'keyword', 'keyword_value', 'target_value'], dtype='object')\n"
keyword_count
wordcloud = WordCloud(background_color='white', max_words=100, max_font_size=80, random_state=42, collocations=False, colormap='Oranges_r').generate(' '.join(train_data[train_data['target'] == 1]['keyword']))
plt.figure(figsize=(10, 10))
plt.title('Major keywords for disaster tweets', fontsize=30)
plt.imshow(wordcloud)
plt.axis('off')

wordcloud = WordCloud(background_color='white', max_words=100, max_font_size=40, collocations=False, colormap='PuOr').generate(' '.join(train_data[train_data['target'] == 0]['keyword']))
print(wordcloud)
plt.figure(figsize=(10, 25))
plt.imshow(wordcloud)
plt.title('Major keywords for non-disaster tweets', fontsize=30)
plt.axis('off')

train_data['location'].value_counts()
train_data['location']
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}', '')
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub('http\\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer('\\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words = [stemmer.stem(w) for w in filtered_words]
    lemma_words = [lemmatizer.lemmatize(w) for w in stem_words]
    return ' '.join(filtered_words)
train_data['location_cleaned'] = train_data['location'].map(lambda s: preprocess(s))
train_data['location_cleaned'].replace({'united states': 'usa', 'world': 'worldwide', 'nyc': 'new york', 'california usa': 'california', 'new york city': 'new york', 'california united states': 'california', 'mumbai': 'india'}, inplace=True)
train_data['location_cleaned'].value_counts().nlargest(20)
train_data['text_cleaned'] = train_data['text'].map(lambda s: preprocess(s))
test_data['text_cleaned'] = test_data['text'].map(lambda s: preprocess(s))
train_text = train_data['text_cleaned']
test_text = test_data['text_cleaned']
train_target = train_data['target']
all_text = train_text.append(test_text)
tfidf_vectorizer = TfidfVectorizer()