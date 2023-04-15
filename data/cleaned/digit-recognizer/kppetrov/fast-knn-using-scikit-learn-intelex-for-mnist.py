import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
x_train = train[train.columns[1:]]
x_test = test
y_train = train[train.columns[0]]
train.head()

def train_predict():
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)