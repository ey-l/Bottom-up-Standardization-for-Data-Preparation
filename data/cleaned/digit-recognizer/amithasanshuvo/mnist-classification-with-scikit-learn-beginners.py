import numpy as np
import pandas as pd
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
train.shape
test.shape
X_train = train.iloc[:, 1:].values.astype('float32')
y_train = train.iloc[:, 0].values.astype('int32')
X_test = test.iloc[:, 1:].values.astype('float32')
y_test = test.iloc[:, 0:].values.astype('int32')
print('X_train: ', X_train)
print('y_train: ', y_train)
print('X_test: ', X_test)
print('y_test: ', y_test)
y_train_5 = y_train == 5
y_test_5 = y_test == 5
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)