import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input2 = pd.read_csv('data/input/titanic/gender_submission.csv')
_input0.head()
_input1.head()
_input1 = pd.get_dummies(_input1, columns=['Embarked'])
_input1.head()
_input0 = pd.get_dummies(_input0, columns=['Embarked'])
_input0.head()
_input1.head()
_input1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
_input0.drop(['Name', 'Ticket', 'Cabin'], axis=1)
_input1['family_size'] = _input1['SibSp'] + _input1['Parch'] + 1
_input1.head()
_input1[['family_size', 'Survived']].groupby(['family_size'], as_index=False).mean().sort_values(by='Survived', ascending=False)
_input0['family_size'] = _input0['SibSp'] + _input0['Parch'] + 1
_input0['family_size'] = _input1['SibSp'] + _input1['Parch'] + 1
_input1.head()
_input1['Sex'] = _input1['Sex'].replace(['male', 'female'], [1, 0], inplace=False)
_input0['Sex'] = _input0['Sex'].replace(['male', 'female'], [1, 0], inplace=False)
_input1.shape
_input1.isnull().sum()
_input0.isnull().sum()
_input1['Age'] = _input1['Age'].fillna(round(_input1['Age'].mean()), inplace=False)
_input0['Age'] = _input0['Age'].fillna(round(_input0['Age'].mean()), inplace=False)
_input1.isnull().sum()
_input0.isnull().sum()
_input0['Fare'] = _input0['Fare'].fillna(round(_input0['Fare'].mean()), inplace=False)
_input1.head()
_input0.head()
_input1['AgeBand'] = pd.cut(_input1['Age'], 5)
_input1[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
_input1.loc[_input1['Age'] <= 16, 'Age'] = 0
_input1.loc[(_input1['Age'] > 16) & (_input1['Age'] <= 32), 'Age'] = 1
_input1.loc[(_input1['Age'] > 32) & (_input1['Age'] <= 48), 'Age'] = 2
_input1.loc[(_input1['Age'] > 48) & (_input1['Age'] <= 64), 'Age'] = 3
_input1.loc[(_input1['Age'] > 64) & (_input1['Age'] <= 80), 'Age'] = 4
_input0.loc[_input0['Age'] <= 16, 'Age'] = 0
_input0.loc[(_input0['Age'] > 16) & (_input0['Age'] <= 32), 'Age'] = 1
_input0.loc[(_input0['Age'] > 32) & (_input0['Age'] <= 48), 'Age'] = 2
_input0.loc[(_input1['Age'] > 48) & (_input0['Age'] <= 64), 'Age'] = 3
_input0.loc[(_input1['Age'] > 64) & (_input0['Age'] <= 80), 'Age'] = 4
_input1[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Age', ascending=True)
_input1.loc[_input1['Fare'] <= 10, 'Fare'] = 0
_input1.loc[(_input1['Fare'] > 10) & (_input1['Fare'] <= 75), 'Fare'] = 1
_input1.loc[_input1['Fare'] > 75, 'Fare'] = 2
_input0.loc[_input0['Fare'] <= 10, 'Fare'] = 0
_input0.loc[(_input0['Fare'] > 10) & (_input0['Fare'] <= 75), 'Fare'] = 1
_input0.loc[_input0['Fare'] > 75, 'Fare'] = 2
_input1.head()
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'family_size']
X = _input1[features]
X_test = _input0[features]
Y = _input1['Survived']
X.head()
X_test.head()
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
models_accuracy = {}
cv = KFold(n_splits=15, random_state=13, shuffle=True)
model = LogisticRegression(solver='liblinear')
scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print(scores)
avg_score = np.round(scores.mean() * 100, 2)
models_accuracy['Logistic Regression'] = avg_score
print('Mean of scores = ', avg_score)
model = SVC(decision_function_shape='ovr')
scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print(scores)
avg_score = np.round(scores.mean() * 100, 2)
models_accuracy['SVM'] = avg_score
print('Mean of scores = {}'.format(np.round(scores.mean() * 100, 2)))
model = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print(scores)
avg_score = np.round(scores.mean() * 100, 2)
models_accuracy['Knn'] = avg_score
print('Mean of scores = {}'.format(np.round(scores.mean() * 100, 2)))
model = RandomForestClassifier(n_estimators=80)
scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print(scores)
avg_score = np.round(scores.mean() * 100, 2)
models_accuracy['Random Forest'] = avg_score
print('Mean of scores = {}'.format(np.round(scores.mean() * 100, 2)))
model = AdaBoostClassifier()
scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print(scores)
avg_score = np.round(scores.mean() * 100, 2)
models_accuracy['Ada Boost'] = avg_score
print('Mean of scores = {}'.format(np.round(scores.mean() * 100, 2)))
model = GradientBoostingClassifier(n_estimators=40)
scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print(scores)
avg_score = np.round(scores.mean() * 100, 2)
models_accuracy['Gradient Boost'] = avg_score
print('Mean of scores = {}'.format(np.round(scores.mean() * 100, 2)))
models_accuracy
main_Model = GradientBoostingClassifier(n_estimators=40)