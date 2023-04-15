import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_df_1 = train
test_df_1 = test
train_df_1.head()
test_df_1.head()
train_df_1.info()
sb.countplot(train_df_1.target)
from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer()
X = vector.fit_transform(train_df_1.text)
Y = train_df_1.target.values
P = vector.transform(test_df_1.text)
from sklearn.model_selection import train_test_split
(xtrain, xtest, ytrain, ytest) = train_test_split(X, Y, test_size=0.2, random_state=101)
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()