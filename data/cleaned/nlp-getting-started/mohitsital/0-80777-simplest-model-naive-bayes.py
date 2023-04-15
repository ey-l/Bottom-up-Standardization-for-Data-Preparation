import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import nltk
import string
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
sno = nltk.stem.SnowballStemmer('english')
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
stopwords = set(stopwords.words('english'))
from sklearn.metrics import classification_report
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print('Done')
dtrain = pd.read_csv('data/input/nlp-getting-started/train.csv')
dtest = pd.read_csv('data/input/nlp-getting-started/test.csv')
submission = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
print('Done')
print(dtrain.columns)
print(dtest.columns)

def cleanhtml(sentence):
    cleantext = re.sub('http\\S+', '', sentence)
    return cleantext

def cleanpunc(sentence):
    cleaned = re.sub('[?|!|\\\'|"|#]', '', sentence)
    cleaned = re.sub('[.|,|)|(|)|\\|/]', ' ', cleaned)
    return cleaned

def cleantxt(data, column):
    str1 = ' '
    final_string = []
    s = ''
    for sent in data[column]:
        filter_sent = []
        rem_html = cleanhtml(sent)
        rem_punc = cleanpunc(rem_html)
        for w in rem_punc.split():
            if w.isalpha() & (len(w) > 2):
                if w.lower() not in stopwords:
                    s = sno.stem(w.lower()).encode('utf8')
                    filter_sent.append(s)
                else:
                    continue
            else:
                continue
        str1 = b' '.join(filter_sent)
        final_string.append(str1)
    data['clean_txt'] = np.array(final_string)
cleantxt(dtrain, 'text')
cleantxt(dtest, 'text')
print(dtrain.columns)
print(dtest.columns)
tf_idf_vect = TfidfVectorizer(ngram_range=(1, 3))
tf_idf_mat = tf_idf_vect.fit_transform(dtrain['clean_txt'].values)
tf_idf_mat_test = tf_idf_vect.transform(dtest['clean_txt'].values)
type(tf_idf_mat)
print(tf_idf_mat.get_shape())
print('done')
target = dtrain['target']
(x, x_test, y, y_test) = train_test_split(tf_idf_mat, target, test_size=0.2, train_size=0.8, random_state=0)