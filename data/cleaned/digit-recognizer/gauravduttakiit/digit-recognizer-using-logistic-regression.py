import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
train.head()
test.head()
y_train = train['label']
X_train = train.drop(labels=['label'], axis=1)
del train
y_train.value_counts()
lr1 = LogisticRegression(solver='lbfgs')