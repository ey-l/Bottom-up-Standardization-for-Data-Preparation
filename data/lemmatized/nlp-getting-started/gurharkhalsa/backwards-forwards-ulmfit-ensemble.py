import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from fastai.text.all import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
import re
import nltk
import string
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def text_preproc(x):
    x = x.lower()
    x = ' '.join([lemmatizer.lemmatize(word) for word in x.split(' ') if word not in stop_words])
    x = x.encode('ascii', 'ignore').decode()
    x = re.sub('https*\\S+', ' ', x)
    x = re.sub('@', ' ', x)
    x = re.sub('amp', '', x)
    x = re.sub('new', '', x)
    x = re.sub(' s ', '', x)
    x = re.sub('#', ' ', x)
    x = re.sub("\\'\\w+", '', x)
    x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x)
    x = re.sub('\\w*\\d+\\w*', '', x)
    x = re.sub('\\s{2,}', ' ', x)
    return x
_input1['text'] = _input1['text'].apply(lambda x: text_preproc(x))
dis = ''
fake = ''
for (i, obj) in _input1.iterrows():
    if obj['target'] == 1:
        dis += obj['text'] + ' '
    elif obj['target'] == 0:
        fake += obj['text'] + ' '
wordcloud_real = WordCloud().generate(dis)
plt.imshow(wordcloud_real)
wordcloud_fake = WordCloud().generate(fake)
plt.imshow(wordcloud_fake)
vectorizer = TfidfVectorizer()
training_set = _input1.sample(frac=0.8)
test_set = _input1.drop(training_set.index)
X = vectorizer.fit_transform(training_set['text'])
test_X = vectorizer.transform(test_set['text'])
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
svc = LinearSVC()
mlp = MLPClassifier(max_iter=3000)
rf_clf = RandomForestClassifier(n_estimators=200, random_state=0, bootstrap=True)
v_clf = VotingClassifier(estimators=[('rf', rf_clf), ('svc', svc), ('mlp', mlp)], voting='hard')