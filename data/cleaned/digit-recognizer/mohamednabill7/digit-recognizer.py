import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('data/input/digit-recognizer/train.csv')
df_test = pd.read_csv('data/input/digit-recognizer/test.csv')
df_train.shape
X = df_train.drop(['label'], axis=1)
y = df_train['label']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, sep='\n')
print(y_train.shape, y_test.shape, sep='\n')
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
svm_clf = SVC()