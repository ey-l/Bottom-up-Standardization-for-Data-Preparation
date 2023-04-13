import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1[:5]
_input0[:2]
for i in _input1.columns:
    print('Column Name  :', i)
    print(_input1[i].unique())
df_train = _input1.drop(['id'], axis=1)
df_test = _input0.drop(['id'], axis=1)
import re
df_train['Hashtag'] = df_train['text'].map(lambda x: re.findall('#(\\w+)', x)).apply(lambda x: ', '.join(x))
df_test['Hashtag'] = df_test['text'].map(lambda x: re.findall('#(\\w+)', x)).apply(lambda x: ', '.join(x))
df_train['@'] = df_train['text'].map(lambda x: re.findall('@(\\w+)', x)).apply(lambda x: ', '.join(x))
df_test['@'] = df_test['text'].map(lambda x: re.findall('@(\\w+)', x)).apply(lambda x: ', '.join(x))

def remove_punctuation(txt):
    import string
    result = txt.translate(str.maketrans('', '', string.punctuation))
    return result

def lower_text(txt):
    return txt.lower()

def remove_no(txt):
    import re
    return re.sub('\\d+', '', txt)

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def removeurl(txt):
    import re
    return re.sub('http?\\S+|www\\.\\S+', '', txt)

def norm(txt):
    x = remove_punctuation(txt)
    x = lower_text(x)
    x = remove_html_tags(x)
    x = remove_no(x)
    x = removeurl(x)
    return x
df_train['text'] = df_train['text'].map(lambda x: norm(x))
df_test['text'] = df_test['text'].map(lambda x: norm(x))
df_train['keyword'] = df_train['keyword'].map(lambda s: s.replace('%20', ' ') if isinstance(s, str) else s)
df_test['keyword'] = df_test['keyword'].map(lambda s: s.replace('%20', ' ') if isinstance(s, str) else s)
df_test
df_train[:5]
train_yes = df_train[df_train['target'] == 1]
train_no = df_train[df_train['target'] == 0]
train_yes[:5]
train_txt_yes = ' '.join((str(i) for i in train_yes['text']))
train_txt_no = ' '.join((str(i) for i in train_no['text']))
test_txt = ' '.join((str(i) for i in df_test['text']))
train_yes['Hashtag'] = train_yes['Hashtag'].replace('^\\s*$', np.nan, regex=True)
train_no['Hashtag'] = train_yes['Hashtag'].replace('^\\s*$', np.nan, regex=True)
train_yes['@'] = train_yes['@'].replace('^\\s*$', np.nan, regex=True)
train_no['@'] = train_yes['@'].replace('^\\s*$', np.nan, regex=True)
print('Total No. of disaster tweets                        :', train_yes['text'].count())
print('Total No. of hashtag present in disaster tweets     :', train_yes['Hashtag'].notnull().sum())
print('Total No. of non-disaster tweets                    :', train_no['text'].count())
print('Total No. of hashtag present in non-disaster tweets :', train_no['Hashtag'].notnull().sum(), '\n')
print('Total No. of disaster tweets                  :', train_yes['text'].count())
print('Total No. of @ present in disaster tweets     :', train_yes['@'].notnull().sum())
print('Total No. of non-disaster tweets              :', train_no['text'].count())
print('Total No. of @ present in non-disaster tweets :', train_no['@'].notnull().sum())

def Token_and_removestopword(txt):
    import nltk
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(txt)
    without_stop_words = []
    for word in words:
        if word not in stop_words:
            without_stop_words.append(word)
    return without_stop_words

def lemmatize_word(tokens, pos='v'):
    import nltk
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, pos=pos) for word in tokens]
    return lemmas
tok_yes = Token_and_removestopword(train_txt_yes)
tok_no = Token_and_removestopword(train_txt_no)
test_tok = Token_and_removestopword(test_txt)
tok_yes[:5]
lem_tok_yes = lemmatize_word(tok_yes)
lem_tok_no = lemmatize_word(tok_no)
lem_test = lemmatize_word(test_tok)
print(len(tok_yes))
print(len(lem_tok_yes))

def top_word_dis(lem_tok, TopN=10):
    fq = nltk.FreqDist(lem_tok)
    rslt = pd.DataFrame(fq.most_common(TopN), columns=['Word', 'Frequency']).set_index('Word')
    plt.style.use('ggplot')
    rslt.plot.bar()
top_word_dis(lem_tok_yes)
top_word_dis(lem_tok_no)
print(lem_test[:10])

def tfidf(train_int, test_int=None, Ngram_min=1, Ngram_max=1):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    def toktotxt(txt_int):
        if isinstance(txt_int[0], list):
            text = txt_int.apply(lambda x: ' '.join((str(i) for i in x)))
        else:
            text = txt_int
        return text
    train_txt = toktotxt(train_int)
    vectorizer = TfidfVectorizer(ngram_range=(Ngram_min, Ngram_max))