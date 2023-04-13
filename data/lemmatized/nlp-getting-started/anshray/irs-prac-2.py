import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import re
import nltk
import spacy
import string
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
print('train dataframe : \n', _input1.head(5))
print(_input1.info())
print('test dataframe : ', _input0.head(5))
print(_input0.info())
print(len(_input1.index))
print(len(_input0.index))
train_df_copy = _input1
_input1 = _input1.drop('target', axis=1)
frames = [_input1, _input0]
_input1 = pd.concat(frames)
_input1['lowered_text'] = _input1['text'].str.lower()
print(_input1['lowered_text'].head(3))
punctuation = string.punctuation
mapping = str.maketrans('', '', punctuation)

def remove_punctuation(in_str):
    return in_str.translate(mapping)
print(_input1['lowered_text'].head(10))
_input1['lowered_text'] = _input1['lowered_text'].apply(lambda x: remove_punctuation(x))
print(_input1['lowered_text'].head(10))
from nltk.corpus import stopwords
stopwords_eng = stopwords.words('english')
print(_input1['lowered_text'].head(10))

def remove_stopwords(in_str):
    new_str = ''
    words = in_str.split()
    for tx in words:
        if tx not in stopwords_eng:
            new_str = new_str + tx + ' '
    return new_str
_input1['lowered_text_stop_removed'] = _input1['lowered_text'].apply(lambda x: remove_stopwords(x))
print(_input1['lowered_text_stop_removed'].head(10))
from collections import Counter
counter = Counter()
for text in _input1['lowered_text_stop_removed']:
    for word in text.split():
        counter[word] += 1
most_cmn_list = counter.most_common(10)
print(type(most_cmn_list), most_cmn_list)
most_cmn_words_list = []
for (word, freq) in most_cmn_list:
    most_cmn_words_list.append(word)
print('Most common words : ', most_cmn_words_list)

def remove_frequent(in_str):
    new_str = ''
    for word in in_str.split():
        if word not in most_cmn_words_list:
            new_str = new_str + word + ' '
    return new_str
_input1['lowered_text_stop_removed_freq_removed'] = _input1['lowered_text_stop_removed'].apply(lambda x: remove_frequent(x))
most_rare_list = counter.most_common()[-10:]
most_rare_words = []
for (word, freq) in most_rare_list:
    most_rare_words.append(word)
print('Most rare words : ', most_rare_words)

def remove_rare(in_text):
    new_text = ''
    for word in in_text.split():
        if word not in most_rare_words:
            new_text = new_text + word + ' '
    return new_text
_input1['lowered_stop_freq_rare_removed'] = _input1['lowered_text_stop_removed_freq_removed'].apply(lambda x: remove_rare(x))
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()

def do_lemmatizing(in_str):
    new_str = ''
    for word in in_str.split():
        new_str = new_str + lem.lemmatize(word) + ' '
    return new_str
_input1['Lemmatized'] = _input1['lowered_stop_freq_rare_removed'].apply(lambda x: do_lemmatizing(x))

def remove_urls(text):
    url_pattern = re.compile('https?://\\S+|www\\.\\S+')
    return url_pattern.sub('', text)

def remove_html(in_str):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub('', in_str)
_input1['urls_removed'] = _input1['Lemmatized'].apply(lambda x: remove_urls(x))
_input1['html_removed'] = _input1['urls_removed'].apply(lambda x: remove_html(x))
chat_words_str = "\nAFAIK=As Far As I Know\nAFK=Away From Keyboard\nASAP=As Soon As Possible\nATK=At The Keyboard\nATM=At The Moment\nA3=Anytime, Anywhere, Anyplace\nBAK=Back At Keyboard\nBBL=Be Back Later\nBBS=Be Back Soon\nBFN=Bye For Now\nB4N=Bye For Now\nBRB=Be Right Back\nBRT=Be Right There\nBTW=By The Way\nB4=Before\nB4N=Bye For Now\nCU=See You\nCUL8R=See You Later\nCYA=See You\nFAQ=Frequently Asked Questions\nFC=Fingers Crossed\nFWIW=For What It's Worth\nFYI=For Your Information\nGAL=Get A Life\nGG=Good Game\nGN=Good Night\nGMTA=Great Minds Think Alike\nGR8=Great!\nG9=Genius\nIC=I See\nICQ=I Seek you (also a chat program)\nILU=ILU: I Love You\nIMHO=In My Honest/Humble Opinion\nIMO=In My Opinion\nIOW=In Other Words\nIRL=In Real Life\nKISS=Keep It Simple, Stupid\nLDR=Long Distance Relationship\nLMAO=Laugh My A.. Off\nLOL=Laughing Out Loud\nLTNS=Long Time No See\nL8R=Later\nMTE=My Thoughts Exactly\nM8=Mate\nNRN=No Reply Necessary\nOIC=Oh I See\nPITA=Pain In The A..\nPRT=Party\nPRW=Parents Are Watching\nROFL=Rolling On The Floor Laughing\nROFLOL=Rolling On The Floor Laughing Out Loud\nROTFLMAO=Rolling On The Floor Laughing My A.. Off\nSK8=Skate\nSTATS=Your sex and age\nASL=Age, Sex, Location\nTHX=Thank You\nTTFN=Ta-Ta For Now!\nTTYL=Talk To You Later\nU=You\nU2=You Too\nU4E=Yours For Ever\nWB=Welcome Back\nWTF=What The F...\nWTG=Way To Go!\nWUF=Where Are You From?\nW8=Wait...\n7K=Sick:-D Laugher\n"
chat_words_expanded_dict = {}
chat_words_list = []
for line in chat_words_str.split('\n'):
    if line != '':
        chat_word = line.split('=')[0]
        chat_word_expanded = line.split('=')[1]
        chat_words_list.append(chat_word)
        chat_words_expanded_dict[chat_word] = chat_word_expanded
chat_words_list = set(chat_words_list)

def convert_chat_words(in_str):
    new_str = ''
    for w in in_str.split():
        if w.upper() in chat_words_list:
            new_str = new_str + chat_words_expanded_dict[w.upper()] + ' '
        else:
            new_str = new_str + w + ' '
    return new_str
_input1['chat_words_coverted'] = _input1['html_removed'].apply(lambda x: convert_chat_words(x))
_input1['spellings_corrected'] = _input1['chat_words_coverted']
print(_input0.shape)
print(_input1.shape)
print(train_df_copy.shape)
_input0 = _input1.iloc[7613:, :]
_input1 = _input1.iloc[:7613, :]
_input1['target'] = train_df_copy['target'].values
_input1.head(5)
count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(_input1['text'])
test_vectors = count_vectorizer.transform(_input0['text'])
print(train_vectors)
print(test_vectors)
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression(random_state=0)