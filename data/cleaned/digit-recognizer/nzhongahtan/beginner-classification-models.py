import numpy as np
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt

import seaborn as sns
fig_dims = (20, 10)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import os
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
train.head(5)
x = train.copy()
x = x.drop(columns=['label'])
y = train['label']
x = x / 255.0
x.iloc[0].max()
(train_x, val_x, train_y, val_y) = train_test_split(x, y, random_state=32)
model = KNeighborsClassifier(n_neighbors=5, weights='distance')