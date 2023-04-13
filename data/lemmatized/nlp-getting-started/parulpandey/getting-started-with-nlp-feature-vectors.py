import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
sentences = ['The weather is sunny', 'The weather is partly sunny and partly cloudy.']
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()