import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
BASE = 'data/input/nlp-getting-started/'
train = pd.read_csv(BASE + 'train.csv')
test = pd.read_csv(BASE + 'test.csv')
sub = pd.read_csv(BASE + 'sample_submission.csv')
tweets = train[['text', 'target']]
tweets.head()
tweets.target.value_counts()
tweets.shape
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def remove_punctuation(text):
    """a function for removing punctuation"""
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
tweets['text'] = tweets['text'].apply(remove_punctuation)
tweets.head(10)
sw = stopwords.words('english')
np.array(sw)

def stopwords(text):
    """a function for removing the stopword"""
    text = [word.lower() for word in word_tokenize(text) if word.lower() not in sw]
    return ' '.join(text)
tweets['text'] = tweets['text'].apply(stopwords)
tweets.head(10)
stemmer = PorterStemmer()

def stemming(text):
    """a function which stems each word in the given text"""
    text = [stemmer.stem(word) for word in word_tokenize(text)]
    return ' '.join(text)
tweets['text'] = tweets['text'].apply(stemming)
tweets.head(10)
vectorizer = CountVectorizer(analyzer='word', binary=True, stop_words='english')