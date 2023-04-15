import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/nlp-getting-started/train.csv')
df.head()
df1 = pd.DataFrame(df['text'])
df1
import re

def date(text):
    pattern = re.compile('\\d{2}[-/.]?\\d{2}[-/.]?\\d{4}')
    text = pattern.findall(text)
    return text
df1['Date'] = df1['text'].apply(lambda x: date(x))
df1

def hashtag(text):
    pattern = re.compile('#\\w*')
    text = pattern.findall(text)
    return text
df1['Hashtag'] = df1['text'].apply(lambda x: hashtag(x))
df1

def mention(text):
    pattern = re.compile('@\\w*')
    text = pattern.findall(text)
    return text
df1['Mention'] = df1['text'].apply(lambda x: mention(x))
df1

def time(text):
    pattern = re.compile('\\d{2}:\\d{2} \\w*')
    text = pattern.findall(text)
    return text
df1['time'] = df1['text'].apply(lambda x: time(x))
df1

def link(text):
    pattern = re.compile('http[s]?://[^\\s]+')
    text = pattern.findall(text)
    return text
df1['URLs'] = df1['text'].apply(lambda x: link(x))
df1

def year(text):
    pattern = re.compile('\\d{4}')
    text = pattern.findall(text)
    return text
df1['Year'] = df1['text'].apply(lambda x: year(x))
df1

def alphanumeric(text):
    pattern = re.compile('^\\w*')
    text = pattern.findall(text)
    return text
df1['Alphanumeric'] = df1['text'].apply(lambda x: alphanumeric(x))
df1

def percentage(text):
    pattern = re.compile('[\\d.\\d]+%')
    text = pattern.findall(text)
    return text
df1['%'] = df1['text'].apply(lambda x: percentage(x))
df1
string = '\nYesterday is 08/03/2022\nToday is 09/03/2022\nTommarow is 10/03/2022'
pattern = re.compile('(\\d{2})\\/(\\d{2})\\/(\\d{4})')
print('Groups :', pattern.findall(string))
print(pattern.sub('\\3-\\2-\\1', string))
string1 = 'Shrikant Shejwal'
pattern = re.compile('(?P<first>\\w+) (?P<last>\\w+)')
print(pattern.findall(string1))
match = pattern.match(string1)
print(match.group('first'))
print(match.group('last'))
print(pattern.sub('\\2 \\1', string1))
print(pattern.sub('\\g<last> \\g<first>', string1))
string3 = 'The energy of god is always working through me.this is a way to keep your mind positive'
pattern = re.compile('way')
print('without boundry =>', pattern.findall(string3))
pattern1 = re.compile('\\bway')
print('with boundry =>', pattern1.findall(string3))
string4 = 'the energy of god is always working through me. the this is a way to keep your mind positive'
pattern = re.compile('the(?=\\senergy)')
print('Positive look ahead "the" word followed by next word energy =>', pattern.findall(string4))
pattern1 = re.compile('the(?!\\senergy)')
print('Negative look ahead "the" word is not followed by next word energy =>', pattern1.findall(string4))