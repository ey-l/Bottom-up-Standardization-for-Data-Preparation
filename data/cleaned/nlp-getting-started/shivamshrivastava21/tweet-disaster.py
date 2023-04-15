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
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
train_df
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
test_df

def text_preprocessing(text):
    review = re.sub('http\\S+', '', text)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review
train_df['processed_text'] = train_df['text'].apply(text_preprocessing)
train_df
test_df['processed_text'] = test_df['text'].apply(text_preprocessing)
test_df
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer()
train_X = train_df['processed_text'].tolist()
test_X = test_df['processed_text'].tolist()
X_train_tf = tf_idf.fit_transform(train_X)
X_train_tf = tf_idf.transform(train_X)
X_test_tf = tf_idf.transform(test_X)
X_test_tf.shape
train_y = train_df['target']
from sklearn.naive_bayes import MultinomialNB