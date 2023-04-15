import pandas as pd
import sklearn as sk
import math
first_sentence = 'Data Science is the sexiest job of the 21st century'
second_sentence = 'machine learning is the key for data science'
first_sentence = first_sentence.split(' ')
second_sentence = second_sentence.split(' ')
total = set(first_sentence).union(set(second_sentence))
print(total)
wordDictA = dict.fromkeys(total, 0)
wordDictB = dict.fromkeys(total, 0)
for word in first_sentence:
    wordDictA[word] += 1
for word in second_sentence:
    wordDictB[word] += 1
pd.DataFrame([wordDictA, wordDictB])

def computeTF(wordDict, doc):
    tfDict = {}
    corpusCount = len(doc)
    for (word, count) in wordDict.items():
        tfDict[word] = count / float(corpusCount)
    return tfDict
tfFirst = computeTF(wordDictA, first_sentence)
tfSecond = computeTF(wordDictB, second_sentence)
tf = pd.DataFrame([tfFirst, tfSecond])
import nltk
from nltk.corpus import stopwords
set(stopwords.words('english'))
filtered_sentence = []
for word in wordDictA:
    if str(word) not in set(stopwords.words('english')):
        filtered_sentence.append(word)
filtered_sentence

def computeIDF(docList):
    idfDict = {}
    N = len(docList)
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for (word, val) in idfDict.items():
        idfDict[word] = math.log10(N / (float(val) + 1))
    return idfDict
idfs = computeIDF([wordDictA, wordDictB])

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for (word, val) in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf
idfFirst = computeTFIDF(tfFirst, idfs)
idfSecond = computeTFIDF(tfSecond, idfs)
idf = pd.DataFrame([idfFirst, idfSecond])
idf
from sklearn.feature_extraction.text import TfidfVectorizer
firstV = 'Data Science is the sexiest job of the 21st century'
secondV = 'machine learning is the key for data science'
vectorize = TfidfVectorizer()
response = vectorize.fit_transform([firstV, secondV])
print(response)