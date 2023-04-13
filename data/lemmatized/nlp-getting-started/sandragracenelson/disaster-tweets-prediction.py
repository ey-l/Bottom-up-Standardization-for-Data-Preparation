import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
_input1.info()
_input1 = _input1.drop(columns=['keyword', 'location'])
_input1.head()
sns.countplot(x='target', data=_input1)
plt.title('Target Distribution')
print(_input1['target'].value_counts())
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
english_stopwords = stopwords.words('english')
', '.join(english_stopwords)
stemmer = SnowballStemmer(language='english')

def tokenize(text):
    return [stemmer.stem(token) for token in word_tokenize(text) if token.isalpha()]
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(_input1['text'])
y_train = _input1['target']
X_test = vectorizer.transform(_input0['text'])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
(X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
lr_model = LogisticRegression(random_state=42)