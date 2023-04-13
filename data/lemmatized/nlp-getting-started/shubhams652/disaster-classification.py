import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
import re
nltk.download('punkt')
nltk.download('wordnet')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
_input0.head()
_input0['text'][0]

def text_preprocessing(text):
    text = re.sub('^RT[\\s]+', '', text)
    text = re.sub('https?:\\/\\/.*[\\r\\n]*', '', text)
    text = re.sub('@[A-Za-z0-9]+', '', text)
    text = re.sub('[^a-zA-Z\\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])
_input1['text'] = _input1['text'].apply(text_preprocessing)
_input0['text'] = _input0['text'].apply(text_preprocessing)
_input0
x = _input1['text']
y = _input1['target']
x_final = _input0['text']
(x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.2, random_state=4, stratify=y)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(x_train).toarray()
X_val = vectorizer.transform(x_val).toarray()
X_final = vectorizer.transform(x_final).toarray()
models = [AdaBoostClassifier(), GradientBoostingClassifier(), LogisticRegression(), RandomForestClassifier(), SVC(), GaussianNB()]
model_names = ['AdaBoostClassifier', 'GradientBoostingClassifier', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Naive Bayes']
accuracies = []
for (model, model_name) in zip(models, model_names):
    pipeline = Pipeline([('model', model)])