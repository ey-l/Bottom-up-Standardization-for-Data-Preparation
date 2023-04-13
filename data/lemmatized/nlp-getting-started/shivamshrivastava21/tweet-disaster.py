import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import re
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input0

def text_preprocessing(text):
    review = re.sub('http\\S+', '', text)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review
_input1['processed_text'] = _input1['text'].apply(text_preprocessing)
_input1
_input0['processed_text'] = _input0['text'].apply(text_preprocessing)
_input0
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer()
train_X = _input1['processed_text'].tolist()
test_X = _input0['processed_text'].tolist()
X_train_tf = tf_idf.fit_transform(train_X)
X_train_tf = tf_idf.transform(train_X)
X_test_tf = tf_idf.transform(test_X)
X_test_tf.shape
train_y = _input1['target']
from sklearn.naive_bayes import MultinomialNB