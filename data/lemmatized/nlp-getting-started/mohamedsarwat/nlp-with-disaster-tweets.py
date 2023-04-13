import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import string
punctation = string.punctuation
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
_input1 = _input1.drop(['id', 'keyword', 'location'], axis=1)
_input1.shape
_input1.columns
_input1.info()
_input1.describe()
_input1[_input1['target'] == 1]['text'].values[0]
_input1[_input1['target'] == 1]['text'].values[1]
print('Number of duplicates in data : {}'.format(len(_input1[_input1.duplicated()])))
print('Duplicated rows before remove them : ')
_input1[_input1.duplicated(keep=False)].sort_values(by='text').head(8)
_input1 = _input1.drop_duplicates(inplace=False)
print('Number of duplicates in data : {}'.format(len(_input1[_input1.duplicated()])))
_input1['target'].value_counts()
plt.figure(figsize=(10, 6))
plt.title('Frequencies of tweets for Disaster')
sns.countplot(x='target', data=_input1)
plt.xlabel('Disaster Type')
Real_Disaster_df = _input1[_input1['target'] == 1]
Real_Disaster_df.head()
Not_Real_Disaster_df = _input1[_input1['target'] == 0]
Not_Real_Disaster_df.head()
Real_Disaster_text = ' '.join(Real_Disaster_df.text.tolist())
wordcloud_true = WordCloud().generate(Real_Disaster_text)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud_true)
plt.axis('off')
plt.title('Word Cloud of Real Disaster news')
plt.tight_layout(pad=0)
Not_Real_Disaster_text = ' '.join(Not_Real_Disaster_df.text.tolist())
wordcloud_true = WordCloud().generate(Not_Real_Disaster_text)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud_true)
plt.axis('off')
plt.title('Word Cloud of Not RealDisaster twittes')
plt.tight_layout(pad=0)

def clean_text(text):
    """
        text: a string 
        return: cleaned string
    """
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and token not in punctation and (len(token) >= 3):
            token = token.lower()
            result.append(token)
    return ' '.join(result)
_input1['text'] = _input1['text'].map(clean_text)
_input1.head()
from sklearn.utils import shuffle
train_df_shuffled = shuffle(_input1)
train_df_shuffled.head()
X = train_df_shuffled['text']
y = train_df_shuffled['target']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test
from sklearn.model_selection import cross_val_score
nb_classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])