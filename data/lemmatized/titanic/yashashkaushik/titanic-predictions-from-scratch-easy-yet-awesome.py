import pandas as pd
import numpy as np
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1
model_data = _input1.drop(['Name', 'Ticket', 'Fare', 'Cabin'], axis=1)
model_data = model_data.dropna(inplace=False)
print(model_data['Sex'].unique())
print(model_data['Pclass'].unique())
print(model_data['Embarked'].unique())
print(model_data['SibSp'].unique())
print(model_data['Parch'].unique())
model_data['Sex'].value_counts()
model_data['Pclass'].value_counts()
model_data['Embarked'].value_counts()
model_data['SibSp'].value_counts()
model_data['Parch'].value_counts()
model_data = pd.get_dummies(model_data, columns=['Sex'])
model_data = pd.get_dummies(model_data, columns=['Pclass'])
model_data = pd.get_dummies(model_data, columns=['Embarked'])
model_data = pd.get_dummies(model_data, columns=['Parch'])
model_data = pd.get_dummies(model_data, columns=['SibSp'])
model_data
y = model_data['Survived']
X = model_data.drop(['Survived', 'PassengerId'], axis=1)
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='liblinear', random_state=0)