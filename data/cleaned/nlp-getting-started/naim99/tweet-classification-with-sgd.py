import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import re
train_data = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_data.head(10)
train_data.dtypes
train_data['text'][11]
import re

def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \\t])|(\\w+:\\/\\/\\S+)|^rt|http.+?', '', elem))
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub('\\d+', '', elem))
    return df
data_clean = clean_text(train_data, 'text', 'text_clean')
data_clean.head()
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
data_clean.head()
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
data_clean['text_tokens'] = data_clean['text_clean'].apply(lambda x: word_tokenize(x))
data_clean.head()
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def word_stemmer(text):
    stem_text = [PorterStemmer().stem(i) for i in text]
    return stem_text
data_clean['text_clean_tokens'] = data_clean['text_tokens'].apply(lambda x: word_stemmer(x))
data_clean.head()
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

def word_lemmatizer(text):
    lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]
    return lem_text
data_clean['text_clean_tokens'] = data_clean['text_tokens'].apply(lambda x: word_lemmatizer(x))
data_clean.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(data_clean['text_clean'], data_clean['target'], random_state=0)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
pipeline_sgd = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('nb', SGDClassifier())])