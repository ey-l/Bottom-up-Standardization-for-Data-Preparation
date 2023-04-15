import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
tweet_data_train = pd.read_csv('data/input/nlp-getting-started/train.csv')
tweet_data_test = pd.read_csv('data/input/nlp-getting-started/test.csv')
tweet_data_train.head()
tweet_data_train.info()
disaster_words = ['forest', 'fire', 'forest fire', 'earthquake', 'landslide', 'typhoon', 'hurricane', 'attack', 'ablaze', 'rains', 'avalanche', 'rescue', 'help', 'hurt', 'god', 'hell', 'died', 'injured', 'succumbed', 'mayhem', 'torrential', 'devastation', 'terror', 'stuck', 'storm', 'unpleasant', 'havoc', 'terrorist', 'tsunami', 'wildfire', 'hailstorm', 'snowfall', 'sinkhole', 'pelting', 'war', 'riot', 'weapon', 'forests', 'fires', 'forest fires', 'earthquakes', 'landslides', 'typhoons', 'hurricanes', 'attacks', 'ablazes', 'rain', 'avalanches', 'rescues', 'terrors', 'storms', 'terrorists', 'tsunamis', 'wildfires', 'hailstorms', 'snowfalls', 'sinkholes', 'peltings', 'riots', 'weapons', '#forest', '#fire', '#forest fire', '#earthquake', '#landslide', '#typhoon', '#hurricane', '#attack', '#ablaze', '#rains', '#avalanche', '#rescue', '#help', '#hurt', '#god', '#hell', '#died', '#injured', '#succumbed', '#mayhem', '#torrential', '#devastation', '#terror', '#stuck', '#storm', '#unpleasant', '#havoc', '#terrorist', '#tsunami', '#wildfires', '#wildfire', '#hailstorm', '#snowfall', '#sinkhole', '#pelting', '#war', '#riot', '#weapon#forests', '#fires', '#forest fires', '#earthquakes', '#landslides', '#typhoons', '#hurricanes', '#attacks', '#ablazes', '#rain', '#avalanches', '#rescues', '#terrors', '#storms', '#terrorists', '#tsunamis', '#wildfires', '#hailstorms', '#snowfalls', '#sinkholes', '#peltings', '#riots', '#weapons']
keywords = list(tweet_data_train['keyword']) + list(tweet_data_test['keyword'])
disaster_words += keywords
disaster_words = list(set(disaster_words))

def make_feature_col(x):
    x['Number of Disaster Words'] = pd.Series()
    tokenizer = RegexpTokenizer('\\w+')
    to_be_removed = set(stopwords.words('english'))
    c = 0
    for i in range(len(x)):
        tokens = tokenizer.tokenize(x['text'][i].lower())
        new_tokens = [word for word in tokens if not word in to_be_removed]
        for j in range(len(new_tokens)):
            if new_tokens[j] in disaster_words:
                c += 1
        x['Number of Disaster Words'].iloc[i] = c
        c = 0
    return x

from wordcloud import WordCloud
unique_string = ' '.join(disaster_words[1:])
wordcloud = WordCloud(width=1000, height=500).generate(unique_string)
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud)
plt.axis('off')

tweet_data_train = make_feature_col(tweet_data_train)
tweet_data_train.head()
new_tweet_data_train = tweet_data_train.loc[:, ['id', 'Number of Disaster Words', 'target']]
new_tweet_data_train
tweet_data_test = make_feature_col(tweet_data_test)
tweet_data_test = tweet_data_test.loc[:, ['id', 'Number of Disaster Words']]
tweet_data_test.head()
X = new_tweet_data_train['Number of Disaster Words']
X = np.array(X).reshape(-1, 1)
y = new_tweet_data_train['target']
dtc = DecisionTreeClassifier(max_depth=10000)
rfc = RandomForestClassifier()