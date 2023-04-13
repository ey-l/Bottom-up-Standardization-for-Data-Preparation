import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
from nltk.corpus import stopwords
re_url = '(?:http|ftp|https)://(?:[\\w_-]+(?:(?:\\.[\\w_-]+)+))(?:[\\w.,@?^=%&:/~+#-]*[\\w@?^=%&/~+#-])?'

def clean_text(text):
    """Make text lowercase, remove reply, remove text in square brackets, remove links, remove user mention,
    remove punctuation, remove numbers and remove words containing numbers."""
    text = text.lower()
    text = re.sub('^rt', '', text)
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub(re_url, '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('@\\w+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text

def get_consecutive_chars(text):
    """ Count how many consecutive chars, consecutive upper chars and consecutive punctuation"""
    result = [(label, sum((1 for _ in group))) for (label, group) in groupby(text)]
    consecutive_chars = 0
    consecutive_chars_upper = 0
    consecutive_punctuations = 0
    for i in result:
        if i[1] > 1:
            if i[0] in string.punctuation:
                consecutive_punctuations += i[1]
            elif i[0].upper() == i[0]:
                consecutive_chars_upper += i[1]
            else:
                consecutive_chars += i[1]
    return {'qtd_consecutive_chars': consecutive_chars, 'qtd_consecutive_chars_upper': consecutive_chars_upper, 'qtd_consecutive_punctuation': consecutive_punctuations}
clean_text('Test 123 of the function clean_text!! https://fake_url/2020')
get_consecutive_chars('test of the function get_consecutive_chars!! lool...')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
idx_train = _input1['id'].values
idx_test = _input0['id'].values
df_full = pd.concat([_input1, _input0], sort=False)
stop_words = stopwords.words('english')
df_full['text_cleaned'] = df_full['text'].apply(clean_text)
df_full['text_cleaned'] = df_full['text_cleaned'].str.split().apply(lambda x: [word for word in x if word not in stop_words]).apply(lambda x: ' '.join(x))
df_full['qnt_words'] = df_full['text_cleaned'].str.split().apply(lambda x: len(x))
df_full['qnt_unique_words'] = df_full['text_cleaned'].str.split().apply(lambda x: len(set(x)))
df_full['qnt_chars'] = df_full['text'].str.len()
df_full['qnt_hashtags'] = df_full['text'].str.findall('#(\\w+)').apply(lambda x: len(x))
df_full['qnt_user_mention'] = df_full['text'].str.findall('@(\\w+)').apply(lambda x: len(x))
df_full['qnt_punctuation'] = df_full['text'].str.replace('[\\w\\s#]+', '').apply(lambda x: len(x))
df_full['qnt_urls'] = df_full['text'].str.findall(re_url).apply(lambda x: len(x))
df_full['mean_chars_words'] = df_full['text'].str.split().apply(lambda x: np.mean([len(w) for w in x]))
df_full['qnt_stop_words'] = df_full['text'].str.split().apply(lambda x: len([w for w in x if w.lower() in stop_words]))
df_full['contains_hashtags'] = df_full['text'].str.findall('#(\\w+)').apply(lambda x: 0 if len(x) == 0 else 1)
df_full['contains_user_mention'] = df_full['text'].str.findall('@(\\w+)').apply(lambda x: 0 if len(x) == 0 else 1)
df_full['contains_punctuation'] = df_full['text'].str.replace('[\\w\\s#]+', '').apply(lambda x: 0 if len(x) == 0 else 1)
df_full['contains_urls'] = df_full['text'].str.findall(re_url).apply(lambda x: len(x))
df_full['is_reply'] = df_full['text'].str.startswith('RT') + 0
df_consecutive = df_full['text'].apply(lambda x: pd.Series(get_consecutive_chars(x)))
for col in df_consecutive.columns:
    df_full[col] = df_consecutive[col]
df_full.columns
df_full.head()
_input1 = df_full[df_full['id'].isin(idx_train)]
_input0 = df_full[df_full['id'].isin(idx_test)]