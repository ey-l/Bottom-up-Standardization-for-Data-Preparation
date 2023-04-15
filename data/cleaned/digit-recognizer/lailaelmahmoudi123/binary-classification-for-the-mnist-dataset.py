import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
mnist_train = pd.read_csv('data/input/digit-recognizer/train.csv')
mnist_test = pd.read_csv('data/input/digit-recognizer/test.csv')
train = mnist_train.copy()
test = mnist_test.copy()
train.shape
test.shape
train.head()
train.tail()
test.head()
test.tail()
train.describe()
print(train.keys())
print(test.keys())
train.isnull().any().any()
(X, y) = (train.drop(labels=['label'], axis=1).to_numpy(), train['label'])
X.shape
X.shape
y.shape
some_digit = X[20]
some_digit_show = plt.imshow(X[20].reshape(28, 28), cmap=mpl.cm.binary)
y[20]
y = y.astype(np.uint8)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1, random_state=42)
y_train_8 = y_train == 8
y_test_8 = y_test == 8
sgd_clf = SGDClassifier(max_iter=1000, random_state=42)