import re
import string
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import plotly.express as px
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm_notebook
import nltk
import gensim
from wordcloud import WordCloud
from nltk.corpus import stopwords
import gensim.downloader as api
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold
warnings.filterwarnings('ignore')
tqdm_notebook.pandas()
nltk.download('stopwords')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
fig = px.histogram(_input1, x='target', nbins=10)
fig.update_layout(template='plotly_dark', title='Binary Classification Counts', width=700, xaxis=dict(dtick=1))
fig.show()
fig = px.histogram(y=_input1['location'].value_counts()[:20], x=_input1['location'].value_counts()[:20].index, color_discrete_sequence=['red'])
fig.update_layout(template='plotly_dark', xaxis_title='Country', yaxis_title='Count', width=1000)
fig.show()
fig = px.histogram(y=_input1['keyword'].value_counts()[:20], x=_input1['keyword'].value_counts()[:20].index, color_discrete_sequence=['orange'])
fig.update_layout(template='plotly_dark', xaxis_title='Keyword', yaxis_title='Count', width=1000)
fig.show()
fig = px.pie(_input1, names='target', title='Pie chart for sentiments of tweets')
fig.update_layout(template='plotly_dark', width=600, height=400)
fig.show()
keyword_dataset = _input1.groupby('location')
keyword_dataset.first()
keyword_dataset = _input1.groupby('keyword')
keyword_dataset.first()

def filter_text(text):
    text = text.lower()
    text = re.sub('http\\S+', '', text)
    text = re.sub('\\x89\\S+', '', text)
    text = re.sub('\\n\\S+', '', text)
    text = re.sub('[0-9]', '', text)
    return text

def PreProcessing(df):
    dataset = df.copy()
    dataset = dataset.fillna('')
    dataset['text'] = dataset['text']
    dataset = dataset.iloc[:, 3:]
    dataset['text'] = dataset['text'].progress_apply(lambda x: filter_text(x))
    return dataset

def remove_punctuation(text):
    if type(text) == float:
        return text
    ans = ''
    for i in text:
        if i not in string.punctuation:
            ans += i
    return ans
dataset = PreProcessing(_input1)
dataset['text'] = dataset['text'].progress_apply(lambda x: remove_punctuation(x))

def generate_unigram(text, n_gram=1):
    words = [word for word in text.split(' ') if word not in set(stopwords.words('english'))]
    temp = zip(*[words[i:] for i in range(0, n_gram)])
    ans = [' '.join(ngram) for ngram in temp]
    result = ''
    for word in ans:
        result = result + word + ' '
    return result
dataset['text'] = dataset['text'].progress_apply(lambda x: generate_unigram(x))
dataset.head()
disaster_tweets_text = dataset.query('target==1').text
concat_disaster_tweets_text = disaster_tweets_text.str.cat(sep=' ')
non_disaster_tweets_text = dataset.query('target==0').text
concat_non_disaster_tweets_text = non_disaster_tweets_text.str.cat(sep=' ')
word_cloud = WordCloud(stopwords=stopwords.words('english'), background_color='white', max_words=100).generate(concat_disaster_tweets_text)
plt.figure(figsize=(20, 8))
plt.imshow(word_cloud, interpolation='bilinear')
plt.title('Disaster tweets')
plt.axis('off')
word_cloud = WordCloud(stopwords=stopwords.words('english'), background_color='white', max_words=100).generate(concat_non_disaster_tweets_text)
plt.figure(figsize=(20, 8))
plt.imshow(word_cloud, interpolation='bilinear')
plt.title('Non-Disaster tweets')
plt.axis('off')

def tokenize_text(text):
    token_list = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            token_list.append(word)
    return token_list
train_tagged = dataset.progress_apply(lambda r: gensim.models.doc2vec.TaggedDocument(tokenize_text(r['text']), [r['target']]), axis=1)

def doc_2_vec_model(train_set):
    train_set = []
    for tag in train_tagged:
        train_set.append(tag)
    dbow_model = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=2, epochs=30)
    dbow_model.build_vocab([x for x in train_set])
    dbow_model.train(train_set, total_examples=len(train_set), epochs=50)
    return (train_set, dbow_model)

def vector_for_learning(model, input_docs):
    sents = input_docs
    (feature_vectors, target) = zip(*[(model.infer_vector(doc.words, steps=20), doc.tags[0]) for doc in tqdm(sents)])
    return (feature_vectors, target)
(train_ds, dbow_model) = doc_2_vec_model(train_tagged)
(x, y) = vector_for_learning(dbow_model, train_ds)
x = np.array(x)
y = np.array(y)

def create_models():
    lr_model = LogisticRegression()
    nb_model = GaussianNB()
    sv_model = SVC()
    ab_model = AdaBoostClassifier()
    mdls = [lr_model, nb_model, sv_model, ab_model]
    mdls_names = ['Logistic Regression', 'Naive Bayes', 'Support Vector Machine', 'AdaBosst Classifier']
    return (mdls, mdls_names)

def models_accuracy(models, models_name, X, Y):
    scores = {}
    grid = [dict(solver=['newton-cg', 'lbfgs', 'liblinear'], penalty=['l2'], C=[100, 10, 1.0, 0.1, 0.01]), dict(var_smoothing=list(np.logspace(0, -9, num=100))), dict(kernel=['poly', 'rbf', 'sigmoid'], C=[50, 10, 1.0, 0.1, 0.01], gamma=['scale']), dict(n_estimators=[10, 50, 100])]
    for (i, model) in tqdm(enumerate(models)):
        rand_search = RandomizedSearchCV(model, grid[i], cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1))