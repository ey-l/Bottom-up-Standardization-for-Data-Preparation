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
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
print('Done')
print(_input1.columns)
print(_input0.columns)

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
cleantxt(_input1, 'text')
cleantxt(_input0, 'text')
print(_input1.columns)
print(_input0.columns)
tf_idf_vect = TfidfVectorizer(ngram_range=(1, 3))
tf_idf_mat = tf_idf_vect.fit_transform(_input1['clean_txt'].values)
tf_idf_mat_test = tf_idf_vect.transform(_input0['clean_txt'].values)
type(tf_idf_mat)
print(tf_idf_mat.get_shape())
print('done')
target = _input1['target']
(x, x_test, y, y_test) = train_test_split(tf_idf_mat, target, test_size=0.2, train_size=0.8, random_state=0)