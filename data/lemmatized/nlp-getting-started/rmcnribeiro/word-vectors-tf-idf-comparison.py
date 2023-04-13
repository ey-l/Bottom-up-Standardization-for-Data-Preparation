import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix
from tqdm import tqdm
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head(3)
_input0.head(3)
_input1.info()
_input0.info()
stop_words_nltk = list(stopwords.words('english'))
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(_input1['text'].values)
csr_matrix(count_train).toarray()
_input1['text'].values[10]
word_tokenize(_input1['text'][10])
from nltk.tokenize import TweetTokenizer

def tweet_tokenize_column(df, column):
    """     
        This function gets the Dataframe and the name of a column (String) containing texts (Strings) and returns
        a list of lists containing the tokenized text. It also turns every token to it's lower form and excludes
        stopwords.
        
        Essa funcao recebe o Dataframe e o nome de uma coluna (String) contendo textos (Strings), e retorna uma lista
        de listas contendo o texto tokenizado. A funcao tambem transforma todas as letras maiusculas em minusculas e 
        exclui stopwords.
        
        Input: Pandas DataFrame, String
        Return: Nested List
    """
    tweet_tokenizer = TweetTokenizer()
    list_sent = [tweet_tokenizer.tokenize(sent) for sent in df[column].values]
    list_sent_no_stop = [[token.lower() for token in sent if token not in stopwords.words('english')] for sent in list_sent]
    return list_sent_no_stop
tokenized_sent_train = tweet_tokenize_column(_input1, 'text')
tokenized_sent_test = tweet_tokenize_column(_input0, 'text')
tokenized_sent_train[:2]
tokenized_sent_test[:2]
tokenized_sent_all = tokenized_sent_train + tokenized_sent_test

def identity_tokenizer(text):
    return text
tfidf_all = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words='english', lowercase=False)
tfidf_all_fit = tfidf_all.fit_transform(tokenized_sent_all)
tfidf_all.get_feature_names()[1000:1002]
tfidf_all_df = pd.DataFrame(tfidf_all_fit.toarray(), columns=tfidf_all.get_feature_names())
tfidf_all_df.head()
tfidf_train_df = tfidf_all_df[:len(_input1)]
tfidf_test_df = tfidf_all_df[len(_input1):]
tfidf_train_df['target_column'] = _input1['target']
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
X = tfidf_train_df.drop('target_column', axis=1)
y = tfidf_train_df['target_column']
clf = LogisticRegression(random_state=16)
scores_logistic = cross_val_score(clf, X, y, cv=5)
scores_logistic.mean()
from sklearn.metrics import accuracy_score