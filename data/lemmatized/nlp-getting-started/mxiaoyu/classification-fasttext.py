import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.tail()
_input0.head()
print(_input1.columns)
print('train_length:', len(_input1), 'test_length:', len(_input0))
print(_input1.keyword.value_counts().sum() / len(_input1))
print(_input1.location.value_counts().sum() / len(_input1))
print('-' * 20)
print(_input1.target.value_counts())
print(len(_input0))
print(_input0.keyword.value_counts().sum())
print(_input0.location.value_counts().sum())
p_k = _input1.groupby('target')['keyword'].agg('count')
p_l = _input1.groupby('target')['location'].agg('count')
(fig, axis) = plt.subplots(1, 2, figsize=(12, 5))
axis[0].bar(p_k.index, p_k.values)
axis[1].bar(p_l.index, p_l.values)

def k_in_t(x1, x2):
    if pd.isnull(x1) == 0:
        if x1 in x2:
            return 1
        else:
            return 0
    else:
        return -1
_input1[['text', 'keyword']].apply(lambda x: k_in_t(x[1], x[0]), axis=1).value_counts()
import fasttext
import jieba
import re
import os
import random
from tqdm import tqdm

def tokenizer(text):
    text = text.strip().lower()
    text = re.sub(re.compile('[a-zA-Z]+://[^\\s]+'), ' ', text)
    su = re.compile('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？°é÷σ―′ⅰⅱ←↓√∶≤≥⊙─━│┏┛▁▌▍▎□△▼▽◆◇○◎●☆〇「〞の㎡“”‘’！[\\]^_`{|}~\\s]+')
    text = re.sub(su, ' ', text)
    text = text.replace('\n', '')
    text = ' '.join(text.split())
    return text

def contact_text(keyword, text, location, target):
    if pd.isnull(keyword):
        keyword = ''
    if pd.isnull(text):
        text = ''
    if pd.isnull(location):
        location = ''
    ss = ' '.join([keyword.strip(), tokenizer(text)])
    sss = '__label__{} '.format(target)
    sss = sss + ss
    return sss

def deal_http(text):
    pattern = re.compile('https?://(www\\.)?(\\w+)(\\.\\w+)(/\\w*)?')
    text = re.sub(pattern, '', text)
    return text

def deal_ascii(text):
    return text.encode('ascii', 'ignore').decode()

def dead_mention(text):
    pat = re.compile('@\\S+')
    return re.sub(pat, '', text)

def remove_email(text):
    pattern = re.compile('[\\w\\.-]+@[\\w\\.-]+\\.\\w+')
    return re.sub(pattern, '', text)

def remove_extra_space(text):
    text = re.sub(' +', ' ', text).strip()
    return text

def replace_us(text):
    text = text.lower()
    text = text.replace('u.s', 'us')
    return text
_input1['text'] = _input1['text'].map(deal_http)
_input1['text'] = _input1['text'].map(deal_ascii)
_input1['text'] = _input1['text'].map(dead_mention)
_input1['text'] = _input1['text'].map(remove_email)
_input1['text'] = _input1['text'].map(remove_extra_space)
_input1['text'] = _input1['text'].map(replace_us)
_input1['text_label'] = _input1[['keyword', 'text', 'location', 'target']].apply(lambda x: contact_text(x[0], x[1], x[2], x[3]), axis=1)
_input1['text_label'].values[250:500]
_input1['text_label'].apply(len).describe()
train_text = _input1['text_label'].tolist()
random.seed(2023)
random.shuffle(train_text)
split_rate = int(np.ceil(len(train_text) * 0.8))
train_x = train_text[:split_rate]
test_x = train_text[split_rate:]
with open('train.txt', 'w', encoding='utf8') as f:
    for sentence in train_x:
        f.write(sentence + '\n')
with open('test.txt', 'w', encoding='utf8') as f:
    for sentence in test_x:
        f.write(sentence + '\n')
fasttext_model = fasttext.train_supervised('train.txt', epoch=80, lr=0.01, dim=128, wordNgrams=3, loss='softmax')
acc = fasttext_model.test('test.txt')
acc

def contact_text2(keyword, text, location):
    if pd.isnull(keyword):
        keyword = ''
    if pd.isnull(text):
        text = ''
    if pd.isnull(location):
        location = ''
    ss = ' '.join([keyword.strip(), tokenizer(text)])
    return ss
_input0['text'] = _input0['text'].map(deal_http)
_input0['text'] = _input0['text'].map(deal_ascii)
_input0['text'] = _input0['text'].map(dead_mention)
_input0['text'] = _input0['text'].map(remove_email)
_input0['text'] = _input0['text'].map(remove_extra_space)
_input0['text_label'] = _input0[['keyword', 'text', 'location']].apply(lambda x: contact_text2(x[0], x[1], x[2]), axis=1)
_input2['target'] = _input0['text_label'].apply(lambda x: int(fasttext_model.predict(str(x))[0][0][-1]))
_input2