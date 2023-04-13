import numpy as np
import pandas as pd
import missingno as mno
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
(_input1['location'].isna().sum(), _input1.shape)
occurences = _input1['target'].value_counts().reset_index().rename(columns={'index': 'Class', 'target': 'Number of Occurences'})
sns.barplot(x=occurences['Class'], y=occurences['Number of Occurences'])
occurences['Percentage(%)'] = occurences['Number of Occurences'] / occurences['Number of Occurences'].sum() * 100
occurences.set_index('Class')
traindata = list(np.array(_input1.iloc[:, 3]))
testdata = list(np.array(_input0.iloc[:, 3]))
y = np.array(_input1.iloc[:, 4]).astype(int)
X_all = traindata + testdata
lentrain = len(traindata)
tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern='\\w{1,}', ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1)
print('Implementing TFIDF to both the test and train data')