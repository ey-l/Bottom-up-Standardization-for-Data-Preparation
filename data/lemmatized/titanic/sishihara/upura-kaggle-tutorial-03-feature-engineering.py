import numpy as np
import pandas as pd
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input2 = pd.read_csv('data/input/titanic/gender_submission.csv')
data = pd.concat([_input1, _input0], sort=False)
data['Sex'] = data['Sex'].replace(['male', 'female'], [0, 1], inplace=False)
data['Embarked'] = data['Embarked'].fillna('S', inplace=False)
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
data['Fare'] = data['Fare'].fillna(np.mean(data['Fare']), inplace=False)
age_avg = data['Age'].mean()
age_std = data['Age'].std()
np.random.randint(age_avg - age_std, age_avg + age_std)
np.random.randint(age_avg - age_std, age_avg + age_std)
data['Age'] = data['Age'].fillna(data['Age'].median(), inplace=False)
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data = data.drop(delete_columns, axis=1, inplace=False)
_input1 = data[:len(_input1)]
_input0 = data[len(_input1):]
y_train = _input1['Survived']
X_train = _input1.drop('Survived', axis=1)
X_test = _input0.drop('Survived', axis=1)
X_train.head()
y_train.head()
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)