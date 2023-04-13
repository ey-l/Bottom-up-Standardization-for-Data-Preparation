import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import plotly
import plotly.express as px
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.info()
_input1.isna().sum()
_input1.isnull().sum() * 100 / len(_input1)
temp = _input1.copy()
temp = temp.dropna()
X = temp.drop(['PassengerId', 'Transported', 'Name'], axis=1)
y = temp['Transported']
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
X = oh.fit_transform(X)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier()