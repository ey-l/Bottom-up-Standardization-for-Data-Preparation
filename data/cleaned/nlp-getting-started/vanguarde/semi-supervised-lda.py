import os
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import numpy as np
from gensim.models.ldamulticore import LdaMulticore
import gensim
from nltk.corpus import stopwords
stops = stopwords.words('english')
import re
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
train.head()

def text_cleaning(texts):
    texts_cleaning = []
    for txt in tqdm(texts):
        url = re.compile('https?://\\S+|www\\.\\S+')
        html = re.compile('<.*?>')
        emoji_pattern = re.compile('[😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿✂-➰Ⓜ-🉑]+', flags=re.UNICODE)
        txt = emoji_pattern.sub('', txt)
        txt = html.sub('', txt)
        txt = url.sub('', txt)
        txt = re.sub('[^A-Za-z\\s]', '', txt)
        texts_cleaning.append(txt.lower())
    return texts_cleaning
text = text_cleaning(train.text.tolist())
from nltk.stem import PorterStemmer
text = [t.split() for t in text]
stemmed_text = []
ps = PorterStemmer()
for sentence in tqdm(text):
    sent = []
    for word in sentence:
        sent.append(ps.stem(word))
    stemmed_text.append(sent)
print(*stemmed_text[5][:20])
print(*text[5][:20])
dictionary = gensim.corpora.Dictionary(stemmed_text)
stopword_ids = map(dictionary.token2id.get, stops)
dictionary.filter_tokens(bad_ids=stopword_ids)
dictionary.filter_extremes(no_below=20, no_above=0.7, keep_n=None)
dictionary.compactify()
bow = [dictionary.doc2bow(line) for line in tqdm(stemmed_text)]
disasters = ['disaster', 'bloodbath', 'collapse', 'crash', 'meltdown', 'doomsday', 'convulsion', 'accident', 'casualty', 'fatality', 'blast', 'catastrophe', 'traffic', 'hybrid', 'engine', 'license', 'tsunami', 'volcano', 'tornado', 'avalanche', 'earthquake', 'blizzard', 'drought', 'bushfire', 'tremor', 'magma', 'twister', 'windstorm', 'cyclone', 'flood', 'fire', 'hailstorm', 'lava', 'lightning', 'hail', 'hurricane', 'seismic', 'erosion', 'whirlpool', 'whirlwind', 'cloud', 'thunderstorm', 'barometer', 'gale', 'blackout', 'gust', 'force', 'volt', 'snowstorm', 'rainstorm', 'storm', 'nimbus', 'violent storm', 'sandstorm', 'fatal', 'cumulonimbus', 'death', 'lost', 'destruction', 'money', 'tension', 'cataclysm', 'damage', 'uproot', 'underground', 'destroy', 'arsonist', 'wind scale', 'arson', 'rescue', 'permafrost', 'fault', 'shelter', 'bomb', 'suicide', 'tragedy', 'weapon']
disasters = [ps.stem(word) for word in disasters]
seed_topics = {}
for word in disasters:
    seed_topics[word] = 0

def create_eta(priors, etadict, ntopics):
    eta = np.full(shape=(ntopics, len(etadict)), fill_value=1)
    for (word, topic) in priors.items():
        keyindex = [index for (index, term) in etadict.items() if term == word]
        if len(keyindex) > 0:
            eta[topic, keyindex[0]] = 10000000.0
    eta = np.divide(eta, eta.sum(axis=0))
    return eta
eta = create_eta(seed_topics, dictionary, 2)
lda_model = LdaMulticore(corpus=bow, id2word=dictionary, num_topics=2, eta=eta, chunksize=2000, passes=10, random_state=42, alpha='symmetric', per_word_topics=True)
for (num, params) in lda_model.print_topics():
    print(f'{num}: {params}\n')