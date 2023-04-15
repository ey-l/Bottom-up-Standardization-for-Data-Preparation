import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
df_train = pd.read_csv('data/input/nlp-getting-started/train.csv')
df_test = pd.read_csv('data/input/nlp-getting-started/test.csv')
df_train.sample(10)
df_train.shape
df_train['target'].value_counts()
df_train.location.fillna('None', inplace=True)
df_train.keyword.fillna('None', inplace=True)
df_train.head()
duplicados_train = df_train['text'].duplicated(keep=False)
duplicados_train.sum()
dup_train = df_train[['text', 'target']][duplicados_train]
dup_train
df_train.shape
df_train = df_train.drop_duplicates('text', keep=False)
df_train.head()
df_train.shape
df_test.head()
df_test.shape
df_test.location.fillna('None', inplace=True)
df_test.keyword.fillna('None', inplace=True)
df_test.head()
duplicados_test = df_test['text'].duplicated(keep=False)
duplicados_test.sum()
dup_test = df_test[['text']][duplicados_test]
dup_test

def minuscula(texto):
    return texto.lower()

def remover_url(texto):
    url = re.compile('https?://\\S+|www\\.\\S+')
    return url.sub('', texto)

def remover_usuario(texto):
    text = re.sub('\\@[A-Za-z0-9]+', '', texto)
    return texto

def remover_emoji(texto):
    emoji_patrones = re.compile('[😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿✂-➰Ⓜ-🉑]+', flags=re.UNICODE)
    return emoji_patrones.sub('', texto)
abreviaturas = {'$': ' dollar ', '€': ' euro ', '4ao': 'for adults only', 'a.m': 'before midday', 'a3': 'anytime anywhere anyplace', 'aamof': 'as a matter of fact', 'acct': 'account', 'adih': 'another day in hell', 'afaic': 'as far as i am concerned', 'afaict': 'as far as i can tell', 'afaik': 'as far as i know', 'afair': 'as far as i remember', 'afk': 'away from keyboard', 'app': 'application', 'approx': 'approximately', 'apps': 'applications', 'asap': 'as soon as possible', 'asl': 'age, sex, location', 'atk': 'at the keyboard', 'ave.': 'avenue', 'aymm': 'are you my mother', 'ayor': 'at your own risk', 'b&b': 'bed and breakfast', 'b+b': 'bed and breakfast', 'b.c': 'before christ', 'b2b': 'business to business', 'b2c': 'business to customer', 'b4': 'before', 'b4n': 'bye for now', 'b@u': 'back at you', 'bae': 'before anyone else', 'bak': 'back at keyboard', 'bbbg': 'bye bye be good', 'bbc': 'british broadcasting corporation', 'bbias': 'be back in a second', 'bbl': 'be back later', 'bbs': 'be back soon', 'be4': 'before', 'bfn': 'bye for now', 'blvd': 'boulevard', 'bout': 'about', 'brb': 'be right back', 'bros': 'brothers', 'brt': 'be right there', 'bsaaw': 'big smile and a wink', 'btw': 'by the way', 'bwl': 'bursting with laughter', 'c/o': 'care of', 'cet': 'central european time', 'cf': 'compare', 'cia': 'central intelligence agency', 'csl': 'can not stop laughing', 'cu': 'see you', 'cul8r': 'see you later', 'cv': 'curriculum vitae', 'cwot': 'complete waste of time', 'cya': 'see you', 'cyt': 'see you tomorrow', 'dae': 'does anyone else', 'dbmib': 'do not bother me i am busy', 'diy': 'do it yourself', 'dm': 'direct message', 'dwh': 'during work hours', 'e123': 'easy as one two three', 'eet': 'eastern european time', 'eg': 'example', 'embm': 'early morning business meeting', 'encl': 'enclosed', 'encl.': 'enclosed', 'etc': 'and so on', 'faq': 'frequently asked questions', 'fawc': 'for anyone who cares', 'fb': 'facebook', 'fc': 'fingers crossed', 'fig': 'figure', 'fimh': 'forever in my heart', 'ft.': 'feet', 'ft': 'featuring', 'ftl': 'for the loss', 'ftw': 'for the win', 'fwiw': 'for what it is worth', 'fyi': 'for your information', 'g9': 'genius', 'gahoy': 'get a hold of yourself', 'gal': 'get a life', 'gcse': 'general certificate of secondary education', 'gfn': 'gone for now', 'gg': 'good game', 'gl': 'good luck', 'glhf': 'good luck have fun', 'gmt': 'greenwich mean time', 'gmta': 'great minds think alike', 'gn': 'good night', 'g.o.a.t': 'greatest of all time', 'goat': 'greatest of all time', 'goi': 'get over it', 'gps': 'global positioning system', 'gr8': 'great', 'gratz': 'congratulations', 'gyal': 'girl', 'h&c': 'hot and cold', 'hp': 'horsepower', 'hr': 'hour', 'hrh': 'his royal highness', 'ht': 'height', 'ibrb': 'i will be right back', 'ic': 'i see', 'icq': 'i seek you', 'icymi': 'in case you missed it', 'idc': 'i do not care', 'idgadf': 'i do not give a damn fuck', 'idgaf': 'i do not give a fuck', 'idk': 'i do not know', 'ie': 'that is', 'i.e': 'that is', 'ifyp': 'i feel your pain', 'IG': 'instagram', 'iirc': 'if i remember correctly', 'ilu': 'i love you', 'ily': 'i love you', 'imho': 'in my humble opinion', 'imo': 'in my opinion', 'imu': 'i miss you', 'iow': 'in other words', 'irl': 'in real life', 'j4f': 'just for fun', 'jic': 'just in case', 'jk': 'just kidding', 'jsyk': 'just so you know', 'l8r': 'later', 'lb': 'pound', 'lbs': 'pounds', 'ldr': 'long distance relationship', 'lmao': 'laugh my ass off', 'lmfao': 'laugh my fucking ass off', 'lol': 'laughing out loud', 'ltd': 'limited', 'ltns': 'long time no see', 'm8': 'mate', 'mf': 'motherfucker', 'mfs': 'motherfuckers', 'mfw': 'my face when', 'mofo': 'motherfucker', 'mph': 'miles per hour', 'mr': 'mister', 'mrw': 'my reaction when', 'ms': 'miss', 'mte': 'my thoughts exactly', 'nagi': 'not a good idea', 'nbc': 'national broadcasting company', 'nbd': 'not big deal', 'nfs': 'not for sale', 'ngl': 'not going to lie', 'nhs': 'national health service', 'nrn': 'no reply necessary', 'nsfl': 'not safe for life', 'nsfw': 'not safe for work', 'nth': 'nice to have', 'nvr': 'never', 'nyc': 'new york city', 'oc': 'original content', 'og': 'original', 'ohp': 'overhead projector', 'oic': 'oh i see', 'omdb': 'over my dead body', 'omg': 'oh my god', 'omw': 'on my way', 'p.a': 'per annum', 'p.m': 'after midday', 'pm': 'prime minister', 'poc': 'people of color', 'pov': 'point of view', 'pp': 'pages', 'ppl': 'people', 'prw': 'parents are watching', 'ps': 'postscript', 'pt': 'point', 'ptb': 'please text back', 'pto': 'please turn over', 'qpsa': 'what happens', 'ratchet': 'rude', 'rbtl': 'read between the lines', 'rlrt': 'real life retweet', 'rofl': 'rolling on the floor laughing', 'roflol': 'rolling on the floor laughing out loud', 'rotflmao': 'rolling on the floor laughing my ass off', 'rt': 'retweet', 'ruok': 'are you ok', 'sfw': 'safe for work', 'sk8': 'skate', 'smh': 'shake my head', 'sq': 'square', 'srsly': 'seriously', 'ssdd': 'same stuff different day', 'tbh': 'to be honest', 'tbs': 'tablespooful', 'tbsp': 'tablespooful', 'tfw': 'that feeling when', 'thks': 'thank you', 'tho': 'though', 'thx': 'thank you', 'tia': 'thanks in advance', 'til': 'today i learned', 'tl;dr': 'too long i did not read', 'tldr': 'too long i did not read', 'tmb': 'tweet me back', 'tntl': 'trying not to laugh', 'ttyl': 'talk to you later', 'u': 'you', 'u2': 'you too', 'u4e': 'yours for ever', 'utc': 'coordinated universal time', 'w/': 'with', 'w/o': 'without', 'w8': 'wait', 'wassup': 'what is up', 'wb': 'welcome back', 'wtf': 'what the fuck', 'wtg': 'way to go', 'wtpa': 'where the party at', 'wuf': 'where are you from', 'wuzup': 'what is up', 'wywh': 'wish you were here', 'yd': 'yard', 'ygtr': 'you got that right', 'ynk': 'you never know', 'zzz': 'sleeping bored and tired'}

def expandir_abreviatura(texto, mapping=abreviaturas):
    texto = ' '.join([mapping[t] if t in mapping else t for t in texto.split(' ')])
    return texto
contracciones_mapeo = {"ain't": 'is not', "aren't": 'are not', "can't": 'cannot', "'cause": 'because', "could've": 'could have', "couldn't": 'could not', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he would', "he'll": 'he will', "he's": 'he is', "how'd": 'how did', "how'd'y": 'how do you', "how'll": 'how will', "how's": 'how is', "I'd": 'I would', "I'd've": 'I would have', "I'll": 'I will', "I'll've": 'I will have', "I'm": 'I am', "I've": 'I have', "i'd": 'i would', "i'd've": 'i would have', "i'll": 'i will', "i'll've": 'i will have', "i'm": 'i am', "i've": 'i have', "isn't": 'is not', "it'd": 'it would', "it'd've": 'it would have', "it'll": 'it will', "it'll've": 'it will have', "it's": 'it is', "let's": 'let us', "ma'am": 'madam', "mayn't": 'may not', "might've": 'might have', "mightn't": 'might not', "mightn't've": 'might not have', "must've": 'must have', "mustn't": 'must not', "mustn't've": 'must not have', "needn't": 'need not', "needn't've": 'need not have', "o'clock": 'of the clock', "oughtn't": 'ought not', "oughtn't've": 'ought not have', "shan't": 'shall not', "sha'n't": 'shall not', "shan't've": 'shall not have', "she'd": 'she would', "she'd've": 'she would have', "she'll": 'she will', "she'll've": 'she will have', "she's": 'she is', "should've": 'should have', "shouldn't": 'should not', "shouldn't've": 'should not have', "so've": 'so have', "so's": 'so as', "this's": 'this is', "that'd": 'that would', "that'd've": 'that would have', "that's": 'that is', "there'd": 'there would', "there'd've": 'there would have', "there's": 'there is', "here's": 'here is', "they'd": 'they would', "they'd've": 'they would have', "they'll": 'they will', "they'll've": 'they will have', "they're": 'they are', "they've": 'they have', "to've": 'to have', "wasn't": 'was not', "we'd": 'we would', "we'd've": 'we would have', "we'll": 'we will', "we'll've": 'we will have', "we're": 'we are', "we've": 'we have', "weren't": 'were not', "what'll": 'what will', "what'll've": 'what will have', "what're": 'what are', "what's": 'what is', "what've": 'what have', "when's": 'when is', "when've": 'when have', "where'd": 'where did', "where's": 'where is', "where've": 'where have', "who'll": 'who will', "who'll've": 'who will have', "who's": 'who is', "who've": 'who have', "why's": 'why is', "why've": 'why have', "will've": 'will have', "won't": 'will not', "won't've": 'will not have', "would've": 'would have', "wouldn't": 'would not', "wouldn't've": 'would not have', "y'all": 'you all', "y'all'd": 'you all would', "y'all'd've": 'you all would have', "y'all're": 'you all are', "y'all've": 'you all have', "you'd": 'you would', "you'd've": 'you would have', "you'll": 'you will', "you'll've": 'you will have', "you're": 'you are', "you've": 'you have'}

def expandir_contraccion(texto, mapping=contracciones_mapeo):
    specials = ['’', '‘', '´', '`']
    for s in specials:
        texto = texto.replace(s, "'")
    texto = ' '.join([mapping[t] if t in mapping else t for t in texto.split(' ')])
    return texto

def remover_tag_html(texto):
    html = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', texto)

def remover_acento(texto):
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return texto

def remover_punto(texto):
    import string
    texto = ''.join([c for c in texto if c not in string.punctuation])
    return texto

def remover_numero(texto):
    texto = ''.join([i for i in texto if not i.isdigit()])
    return texto

def remover_espacio_extra(texto):
    import re
    pattern = '^\\s*|\\s\\s*'
    return re.sub(pattern, ' ', texto).strip()

def remover_no_alfabeto(texto):
    return ' '.join([i for i in texto.split() if i.isalpha() == True])

def remover_stop_word(texto):
    return ' '.join([word for word in word_tokenize(texto) if not word in stopwords.words('english')])

def lematizar(texto):
    lemma = WordNetLemmatizer()
    return ' '.join([lemma.lemmatize(word) for word in word_tokenize(texto)])

def preprocesar_df(df, col_name, clean_col_name):
    df[clean_col_name] = df[col_name].apply(lambda x: minuscula(x)).apply(lambda x: remover_url(x)).apply(lambda x: remover_usuario(x)).apply(lambda x: remover_emoji(x)).apply(lambda x: expandir_abreviatura(x)).apply(lambda x: expandir_contraccion(x)).apply(lambda x: remover_tag_html(x)).apply(lambda x: remover_acento(x)).apply(lambda x: remover_punto(x)).apply(lambda x: remover_numero(x)).apply(lambda x: remover_espacio_extra(x)).apply(lambda x: remover_no_alfabeto(x)).apply(lambda x: remover_stop_word(x)).apply(lambda x: lematizar(x))
preprocesar_df(df_train, 'text', 'texto_preprocesado')
df_train.head(30)
df_train.shape
x_train_original = df_train['text']
x_train_preprocesado = df_train['texto_preprocesado']
y_train = df_train['target']
preprocesar_df(df_test, 'text', 'texto_preprocesado')
df_test.head()
df_test.shape
x_test_original = df_test['text']
x_test_preprocesado = df_test['texto_preprocesado']
parametros_bow = {'vectorizador__strip_accents': 'unicode', 'vectorizador__ngram_range': [(1, 1), (2, 2), (1, 2)], 'vectorizador__max_df': [10, 20, 30], 'vectorizador__binary': [False, True]}
parametros_tfidf = {'vectorizador__strip_accents': 'unicode', 'vectorizador__ngram_range': [(1, 1), (2, 2), (1, 2)], 'vectorizador__max_df': [10, 20, 30], 'vectorizador__binary': [False, True], 'vectorizador__use_idf': [True, False]}
pipeline_bow_RR = Pipeline([('vectorizador', CountVectorizer()), ('clf', RidgeClassifier())])
hiperparametros_1 = {'vectorizador__strip_accents': ['unicode'], 'vectorizador__ngram_range': [(1, 1), (2, 2), (1, 2)], 'vectorizador__min_df': [10, 20, 30], 'vectorizador__binary': [False, True], 'clf__alpha': [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], 'clf__normalize': [False, True], 'clf__random_state': [42]}
clf_RR_bow_original = GridSearchCV(pipeline_bow_RR, hiperparametros_1, cv=5, n_jobs=-1, verbose=2)