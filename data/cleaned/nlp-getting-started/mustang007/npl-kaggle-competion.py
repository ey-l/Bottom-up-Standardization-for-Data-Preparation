import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataset = pd.read_csv('data/input/nlp-getting-started/train.csv', usecols=['text', 'target'])
dataset.head(20)
test_data = pd.read_csv('data/input/nlp-getting-started/test.csv', usecols=['text'])
dataset.groupby(['target']).count().plot.bar()
dataset.describe()
dataset.describe(include='O')
dataset.drop_duplicates(subset='text', keep='first', inplace=True)
import string
import nltk
from nltk.corpus import stopwords
stopword = stopwords.words('english')

def text_preprocessing(texts):
    tex = texts.strip()
    texts_word = [word for word in tex.split() if '@' not in word]
    tex = ' '.join(texts_word)
    texts_word = [word for word in tex.split() if '#' not in word]
    tex = ' '.join(texts_word)
    texts_word = [word for word in tex.split() if 'www.' not in word]
    tex = ' '.join(texts_word)
    texts_word = [word for word in tex.split() if 'http' not in word]
    tex = ' '.join(texts_word)
    texts_word = [word for word in tex if word not in string.punctuation]
    tex = ''.join(texts_word)
    texts_word = [word for word in tex.split() if word not in stopword]
    tex = ' '.join(texts_word)
    texts_word = [word for word in tex.split() if word.isalpha()]
    tex = ' '.join(texts_word)
    texts_word = [word.lower() for word in tex.split()]
    tex = ' '.join(texts_word)
    tex = tex.strip()
    return tex.split()
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
dataset['text'] = dataset['text'].astype('str')
test_data['text'] = test_data['text'].astype('str')