import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.head()
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
from sklearn.ensemble import RandomForestClassifier
y = _input1['Survived']
features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = pd.get_dummies(_input1[features])
X_test = pd.get_dummies(_input0[features])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)