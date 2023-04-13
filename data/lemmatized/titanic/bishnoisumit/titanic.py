import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input2 = pd.read_csv('data/input/titanic/gender_submission.csv')
_input1
_input1.info()
_input0.describe()
print('Number of null values in different columns are: ')
print('--------------------------------------------------')
print(_input1.isna().sum())
print('--------------------------------------------------')
_input1.loc[_input1['Cabin'].isna() == False, 'Cabin'] = 1
_input1.loc[_input1['Cabin'].isna() == True, 'Cabin'] = 0
_input1.loc[_input1['Age'].isna() == True, 'Age'] = _input1['Age'].mean()
_input1['Embarked'].mode()
_input1.loc[_input1['Embarked'].isna() == True, 'Embarked'] = 'S'
print('Number of null values in different columns are: ')
print('--------------------------------------------------')
print(_input1.isna().sum())
print('--------------------------------------------------')
_input1 = _input1.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=False)
_input1.loc[_input1['Sex'] == 'female', 'Sex'] = 1
_input1.loc[_input1['Sex'] == 'male', 'Sex'] = 0
ohe = OneHotEncoder(sparse=False, handle_unknown='error', drop='first')
ohe_df = pd.DataFrame(ohe.fit_transform(_input1[['Embarked']]))
ohe_df.columns = ohe.get_feature_names(['Embarked'])
ohe_df.head()
_input1 = pd.concat([_input1, ohe_df], axis=1)
_input1 = _input1.drop(['Embarked'], axis=1, inplace=False)
ColumnToScale = ['Age', 'Fare']
_input1[ColumnToScale] = MinMaxScaler().fit_transform(_input1[ColumnToScale])
_input1['Fare'] = _input1['Fare'] * 10
_input1
_input1.describe()
_input1.shape
X_train = _input1.iloc[:, 1:10].values
Y_train = _input1.iloc[:, 0].values
classifier = SVC(kernel='rbf', random_state=1)