import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from collections import Counter, defaultdict
import altair as alt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import f1_score
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
stop_words = set(stopwords.words('english'))

def text_cleaner(text, num):
    """
        Text cleaner does the following
        1. Lowercase text
        2. Removes non text from raw reviews
        3. Substitutes not alphanumeric characters
        4. Correct words using the contractions mapping dictionary
        5. Removes Junk characters generated after cleaning
        6. Remove stop words if num=0 that means for review only not for summary
        
        Parameters: String, Number
        Returns: String
    """
    newString = text.lower()
    newString = BeautifulSoup(newString, 'lxml').text
    newString = re.sub('\\([^)]*\\)', '', newString)
    newString = re.sub('http', '', newString)
    newString = re.sub('"', '', newString)
    newString = re.sub("'s\\b", '', newString)
    newString = re.sub('[^a-zA-Z]', ' ', newString)
    newString = re.sub('[m]{2,}', 'mm', newString)
    if num == 0:
        tokens = [w for w in newString.split() if not w in stop_words]
    else:
        tokens = newString.split()
    long_words = []
    for i in tokens:
        if len(i) > 1:
            long_words.append(i)
    return ' '.join(long_words).strip()
cleaned_text_train = []
for t in _input1['text']:
    cleaned_text_train.append(text_cleaner(t, 1))
cleaned_text_test = []
for t in _input0['text']:
    cleaned_text_test.append(text_cleaner(t, 1))
_input1['text'] = cleaned_text_train
_input0['text'] = cleaned_text_test
_input1['keyword'] = _input1['keyword'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', str(x)))
_input1['keyword'] = _input1['keyword'].apply(lambda x: lemmatizer.lemmatize(x.lower()))
_input0['keyword'] = _input0['keyword'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', str(x)))
_input0['keyword'] = _input0['keyword'].apply(lambda x: lemmatizer.lemmatize(x.lower()))
_input1['location'] = _input1['location'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', str(x)))
_input0['location'] = _input0['location'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', str(x)))
_input1['full_text'] = _input1['keyword'] + ' ' + _input1['location'] + ' ' + _input1['text']
_input0['full_text'] = _input0['keyword'] + ' ' + _input0['location'] + ' ' + _input0['text']
_input1['full_text'] = _input1['full_text'].apply(lambda x: re.sub('nan', ' ', str(x)))
_input0['full_text'] = _input0['full_text'].apply(lambda x: re.sub('nan', ' ', str(x)))
temp0 = _input1.target.value_counts().reset_index(name='count')
alt.Chart(temp0, title='Non Disaster vs Disaster tweet count').mark_bar().encode(x='index:O', y='count', color='index:O').properties(width=400)
no_missing = _input1[(_input1['keyword'] != 'nan') & (_input1['location'] != 'nan')]
temp1 = no_missing['keyword'].value_counts().head(20).reset_index(name='count')
alt.Chart(temp1, title='Top 20 keywords in tweets').mark_bar().encode(alt.X('index', axis=alt.Axis(labelAngle=-45)), y='count').properties(width=700).configure_axis(labelFontSize=15, titleFontSize=15)
temp2 = no_missing['location'].value_counts().head(20).reset_index(name='count')
alt.Chart(temp2, title='Top 20 locations of tweets').mark_bar().encode(alt.X('index', axis=alt.Axis(labelAngle=-45)), y='count').properties(width=700).configure_axis(labelFontSize=15, titleFontSize=15)
temp3 = no_missing.groupby(['keyword', 'location']).size().nlargest(20).reset_index(name='count')
alt.Chart(temp3, title='Top location and keyword combinations').mark_circle(size=60).encode(alt.X('location', axis=alt.Axis(labelAngle=-45)), y='count', color='keyword', tooltip=['location', 'keyword', 'count']).properties(width=700).interactive().configure_axis(labelFontSize=15, titleFontSize=15)
temp4 = no_missing.groupby(['keyword', 'target']).size().nlargest(40).reset_index(name='count')
alt.Chart(temp4, title='How good a keyword is indicator of disaster?').mark_bar().encode(x='target:O', y='count', color='target:N', column='keyword').properties(width=40)
temp5 = no_missing.groupby(['location', 'target']).size().nlargest(30).reset_index(name='count')
alt.Chart(temp5, title='How good a location is indicator of disaster?').mark_bar().encode(x='target:O', y='count', color='target:N', column='location').properties(width=40)
temp6 = no_missing[no_missing['target'] == 1]['keyword'].value_counts().head(20).reset_index(name='count')
alt.Chart(temp6, title='Top keywords present in Disaster tweets').mark_bar().encode(alt.X('index', axis=alt.Axis(labelAngle=-45)), y='count').properties(width=700).configure_axis(labelFontSize=15, titleFontSize=15)
list_corpus = _input1['full_text'].tolist()
list_labels = _input1['target'].tolist()
(X_train, X_val, y_train, y_val) = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=40)
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english', binary=True, lowercase=True)
train_vectors = vectorizer.fit_transform(X_train)
val_vectors = vectorizer.transform(X_val)
logreg = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', n_jobs=-1, random_state=40)