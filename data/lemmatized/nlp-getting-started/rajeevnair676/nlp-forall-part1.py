import nltk
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
sample_text = 'FSD Beta 10.69 started rolling out to Tesla owners last night. This build is a big step forward! 10.69.1 probably end of week with wider release.'
word_tokenize(sample_text)
sent_tokenize(sample_text)

def generate_N_grams(text, ngram):
    words = [word for word in text.split(' ')]
    temp = zip(*[words[i:] for i in range(0, ngram)])
    ans = [' '.join(ngram) for ngram in temp]
    return ans
text = 'I hear they are calling for thunderstorms all weekend.'
generate_N_grams(text, 1)
generate_N_grams(text, 2)
generate_N_grams(text, 3)
import numpy as np
from collections import defaultdict
doc1 = 'Man bites the dog'
doc2 = 'Dog bites man'
doc3 = 'Man likes dog'
doc4 = 'Dog loves man but the man hates the dog'
vocab = [' '.join((str(word).lower() for doc in [doc1, doc2, doc3, doc4] for word in doc.split()))]
print(vocab)
word2idx = {}
for (idx, word) in enumerate(set(vocab[0].split())):
    word2idx[word] = idx
print(word2idx)
corpus = list(set(vocab[0].split()))
print(corpus)

def one_hot_encoding(text):
    """Function to build a one hot encoding of words in the text
    
    #Arguments
       text: The string text that we need to convert to one hot encoding
       
    #Returns
       one_hot : Returns the one hot representation of the words in the sentence
    """
    one_hot = []
    for word in text.split():
        temp = [0] * len(corpus)
        if word in corpus:
            temp[word2idx[word]] = 1
        one_hot.append(temp)
    return one_hot
df_onehot = pd.DataFrame(one_hot_encoding(doc1.lower()))
df_onehot.columns = corpus
df_onehot.style.background_gradient(cmap='Blues')

def bag_of_words(sent):
    """Function to build BoW representation of the input text
    
    # Argument:
        sent: The input sentence to be converted
    # Returns
        vec : The BoW representation of the input sentence
    """
    count_dict = defaultdict(int)
    vec = np.zeros(len(corpus))
    for item in sent.split():
        count_dict[item] += 1
    for (key, item) in count_dict.items():
        vec[word2idx[key]] = item
    return vec
bow_vectors = []
for doc in [doc1, doc2, doc3, doc4]:
    bow_vec = bag_of_words(doc.lower())
    bow_vectors.append(bow_vec)
bow_df = pd.DataFrame(bow_vectors)
bow_df.columns = corpus
bow_df.style.background_gradient(cmap='Blues')
from collections import Counter
import numpy as np
sentence = ['this is the first sentence', 'this is the second sentence', 'the third sentence is this one', 'this is the fourth sentence and it is fantastic']
total_docs = len(sentence)
vocab = ' '.join((str(word).lower() for sent in sentence for word in sent.split()))
word_count = Counter(vocab.split())
word_set = list(set(vocab.split()))
print(word_set)
word2idx = {}
for (idx, word) in enumerate(word_set):
    word2idx[word] = idx
print(word2idx)

def term_freq(sentence, word):
    """Function to compute the term frequency given a sentence and word
    
    #Arguments
        sentence: The sentence or the document that we are considering
        word: The word in the sentence for which we need to calculate term frequency
    #Returns
        Returns the term frequency
    """
    len_sen = len(sentence.split())
    frequency = len([token for token in sentence.split() if word == token])
    return frequency / len_sen

def idf(word):
    """Function to calculate the IDF of the document corpus
    
    #Arguments:
        word : The word for which we need to calculate the IDF
    #Returns:
        The IDF value
    """
    doc_len = len(sentence)
    doc_count = 0
    for sent in sentence:
        if word in sent:
            doc_count += 1
    return np.log(doc_len / doc_count) + 1
full_vec = []
for sent in sentence:
    vec = np.zeros(len(word_set))
    for word in sent.split():
        tfidf = term_freq(sent, word) * idf(word)
        vec[word2idx[word]] = tfidf
    full_vec.append(vec)
tfidf_df = pd.DataFrame(full_vec, index=['Sent1', 'Sent2', 'Sent3', 'Sent4'])
tfidf_df.columns = word_set
tfidf_df.style.background_gradient(cmap='Purples')