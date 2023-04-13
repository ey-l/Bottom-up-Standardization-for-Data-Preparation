import numpy as np
import pandas as pd
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1.head()
_input1.describe()
null_columns = _input1.columns[_input1.isnull().any()]
_input1[null_columns].isnull().sum()
y = _input1.Survived
features = ['Fare', 'Pclass']
X = _input1[features]
from sklearn.linear_model import LogisticRegression
titanic_mdl = LogisticRegression(random_state=0)