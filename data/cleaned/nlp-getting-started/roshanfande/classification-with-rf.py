import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub("\\'s", '', string)
    string = re.sub("\\'ve", '', string)
    string = re.sub("n\\'t", '', string)
    string = re.sub("\\'re", '', string)
    string = re.sub("\\'d", '', string)
    string = re.sub("\\'ll", '', string)
    string = re.sub(',', '', string)
    string = re.sub('!', ' ! ', string)
    string = re.sub('\\(', '', string)
    string = re.sub('\\)', '', string)
    string = re.sub('\\?', '', string)
    string = re.sub("'", '', string)
    string = re.sub("[^A-Za-z0-9(),!?\\'\\`]", ' ', string)
    string = re.sub('[0-9]\\w+|[0-9]', '', string)
    string = re.sub('\\s{2,}', ' ', string)
    return string.strip().lower()
data = pd.read_csv('data/input/nlp-getting-started/train.csv')
x = data['text'].tolist()
y = data['target'].tolist()
for (index, value) in enumerate(x):
    print('processing data:', index)
    x[index] = ' '.join([Word(word).lemmatize('v') for word in clean_str(value).split()])
vect = TfidfVectorizer(stop_words='english', min_df=2)
X = vect.fit_transform(x)
Y = np.array(y)
print('no of features extracted:', X.shape[1])
(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, random_state=42)
print('train size:', X_train.shape)
print('test size:', X_test.shape)
model = RandomForestClassifier(n_estimators=300, max_depth=150, n_jobs=1)