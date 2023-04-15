import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
train_data = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_data.head()
np.unique(train_data.target, return_counts=True)
target_count = train_data.groupby('target').size().reset_index(name='counts')
plt.bar(target_count.target, target_count.counts)
plt.xticks([0, 1], labels=['Not disaster tweets', 'disaster tweets'])
plt.title('Target Distribution')


def preprocess(reviews):
    tokenizer = RegexpTokenizer('\\w+')
    review = str(reviews)
    review = review.lower()
    review = review.replace('<br /><br />', '')
    tokens = tokenizer.tokenize(review)
    stop_words = set(stopwords.words('english'))
    stopwords_removed = [i for i in tokens if i not in stop_words]
    ps = PorterStemmer()
    stem_text = [ps.stem(i) for i in stopwords_removed]
    cleaned_reviews = ' '.join(stem_text)
    return cleaned_reviews
X = train_data.text.apply(preprocess)
y = train_data.target
(X_train, X_test, y_train, y_test) = train_test_split(np.array(X), y, test_size=0.2, random_state=10)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
model1 = Pipeline([('tfidf', TfidfVectorizer()), ('mnb', MultinomialNB())])
model2 = Pipeline([('c_vec', CountVectorizer()), ('mnb', MultinomialNB())])
model3 = Pipeline([('tfidf', TfidfVectorizer()), ('bern', BernoulliNB())])
model4 = Pipeline([('c_vec', CountVectorizer()), ('bern', BernoulliNB())])