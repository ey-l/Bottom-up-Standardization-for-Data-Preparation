import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
passangerId = _input0['PassengerId']
final_dataframe = _input1[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
final_dataframe = final_dataframe.dropna()
final_dataframe.head()
final_dataframe['Sex'] = final_dataframe['Sex'].replace(to_replace=final_dataframe['Sex'].unique(), value=[1, 0])
final_dataframe = pd.get_dummies(final_dataframe, drop_first=True)
train_y = final_dataframe['Survived']
train_x = final_dataframe[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
from sklearn.model_selection import train_test_split
(train_data, val_data, train_target, val_target) = train_test_split(train_x, train_y, train_size=0.8)
model = GaussianNB()