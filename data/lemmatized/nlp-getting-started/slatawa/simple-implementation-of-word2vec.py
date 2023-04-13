import numpy as np
import pandas as pd
import warnings
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
print(_input1.head(5))
print(_input1.info())

def clean_text(text):
    import re
    text = text.lower()
    text = re.sub("i'm", 'i am', text)
    text = re.sub("you'll", 'you will', text)
    text = re.sub("i'll", 'i will', text)
    text = re.sub("she'll", 'she will', text)
    text = re.sub("he'll", 'he will', text)
    text = re.sub("he's", 'he is', text)
    text = re.sub("she's", 'she is', text)
    text = re.sub("that's", 'that is', text)
    text = re.sub("what's", 'what is', text)
    text = re.sub("where's", 'where is', text)
    text = re.sub("there's", 'there is', text)
    text = re.sub("here's", 'here is', text)
    text = re.sub("who's", 'who is', text)
    text = re.sub("how's", 'how is', text)
    text = re.sub("\\'ll", ' will', text)
    text = re.sub("\\'ve", ' have', text)
    text = re.sub("\\'re", ' are', text)
    text = re.sub("\\'d", ' would', text)
    text = re.sub("can't", 'cannot', text)
    text = re.sub("won't", 'will not', text)
    text = re.sub("don't", 'do not', text)
    text = re.sub("shouldn't", 'should not', text)
    text = re.sub("n't", ' not', text)
    text = re.sub('   ', ' ', text)
    return text
_input1['clean_text'] = _input1['text'].apply(clean_text)
_input0['clean_text'] = _input0['text'].apply(clean_text)

def massage_text(text):
    import re
    from nltk.corpus import stopwords
    tweet = re.sub('[^a-zA-Z]', ' ', text)
    tweet = tweet.lower()
    tweet = tweet.split()
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    tweet = [lem.lemmatize(word) for word in tweet if word not in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    return tweet
    print('--here goes nothing')
    print(text)
    print(tweet)
_input1['clean_text'] = _input1['text'].apply(massage_text)
_input0['clean_text'] = _input0['text'].apply(massage_text)
_input1.iloc[0:10][['text', 'clean_text']]
from nltk.tokenize import word_tokenize
_input1['tokens'] = _input1['clean_text'].apply(lambda x: word_tokenize(x))
_input0['tokens'] = _input0['clean_text'].apply(lambda x: word_tokenize(x))
_input1
import gensim

def fn_pre_process_data(doc):
    for rec in doc:
        yield gensim.utils.simple_preprocess(rec)
corpus = list(fn_pre_process_data(_input1['clean_text']))
corpus += list(fn_pre_process_data(_input0['clean_text']))
from gensim.models import Word2Vec
print('initiated ...')
wv_model = Word2Vec(corpus, size=150, window=3, min_count=2)
wv_model.train(corpus, total_examples=len(corpus), epochs=10)

def get_word_embeddings(token_list, vector, k=150):
    if len(token_list) < 1:
        return np.zeros(k)
    else:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in token_list]
    sum = np.sum(vectorized, axis=0)
    return sum / len(vectorized)

def get_embeddings(tokens, vector):
    embeddings = tokens.apply(lambda x: get_word_embeddings(x, wv_model))
    return list(embeddings)
train_embeddings = get_embeddings(_input1['tokens'], wv_model)
test_embeddings = get_embeddings(_input0['tokens'], wv_model)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
lr_model = LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
grid_search_model = GridSearchCV(lr_model, param_grid=grid_values, cv=3)