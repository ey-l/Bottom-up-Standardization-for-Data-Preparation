import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sub = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.tail()
test.head()
print(train.columns)
print('train_length:', len(train), 'test_length:', len(test))
print(train.keyword.value_counts().sum() / len(train))
print(train.location.value_counts().sum() / len(train))
print('-' * 20)
print(train.target.value_counts())
print(len(test))
print(test.keyword.value_counts().sum())
print(test.location.value_counts().sum())
p_k = train.groupby('target')['keyword'].agg('count')
p_l = train.groupby('target')['location'].agg('count')
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
train[['text', 'keyword']].apply(lambda x: k_in_t(x[1], x[0]), axis=1).value_counts()
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
train['text'] = train['text'].map(deal_http)
train['text'] = train['text'].map(deal_ascii)
train['text'] = train['text'].map(dead_mention)
train['text'] = train['text'].map(remove_email)
train['text'] = train['text'].map(remove_extra_space)
train['text'] = train['text'].map(replace_us)
train['text_label'] = train[['keyword', 'text', 'location', 'target']].apply(lambda x: contact_text(x[0], x[1], x[2], x[3]), axis=1)
train['text_label'].values[250:500]
train['text_label'].apply(len).describe()
train_text = train['text_label'].tolist()
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
test['text'] = test['text'].map(deal_http)
test['text'] = test['text'].map(deal_ascii)
test['text'] = test['text'].map(dead_mention)
test['text'] = test['text'].map(remove_email)
test['text'] = test['text'].map(remove_extra_space)
test['text_label'] = test[['keyword', 'text', 'location']].apply(lambda x: contact_text2(x[0], x[1], x[2]), axis=1)
sub['target'] = test['text_label'].apply(lambda x: int(fasttext_model.predict(str(x))[0][0][-1]))

sub