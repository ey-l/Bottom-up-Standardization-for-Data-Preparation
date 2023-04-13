import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.shape
_input0.shape
_input1.head()
_input0.head()
_input1.info()
_input1.describe()
_input1.PassengerId.nunique() / _input1.shape[0]
_input1.Transported.value_counts()
_input1.isnull().sum()
null_value = _input0.isnull().sum()
null_value
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(method='ffill', inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(method='ffill', inplace=False)
_input1['Cabin'] = _input1['Cabin'].fillna(method='ffill', inplace=False)
_input1['Destination'] = _input1['Destination'].fillna(method='ffill', inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input1['VIP'] = _input1['VIP'].fillna(method='ffill', inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].mean(), inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].mean(), inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].mean(), inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].mean(), inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].mean(), inplace=False)
_input1['Name'] = _input1['Name'].fillna(method='ffill', inplace=False)
_input1.isnull().sum()
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(method='ffill', inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(method='ffill', inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna(method='ffill', inplace=False)
_input0['Destination'] = _input0['Destination'].fillna(method='ffill', inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean(), inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(method='ffill', inplace=False)
_input0['RoomService'] = _input0['RoomService'].fillna(_input0['RoomService'].mean(), inplace=False)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(_input0['FoodCourt'].mean(), inplace=False)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(_input0['ShoppingMall'].mean(), inplace=False)
_input0['Spa'] = _input0['Spa'].fillna(_input0['Spa'].mean(), inplace=False)
_input0['VRDeck'] = _input0['VRDeck'].fillna(_input0['VRDeck'].mean(), inplace=False)
_input0['Name'] = _input0['Name'].fillna(method='ffill', inplace=False)
_input0.isnull().sum()
plt.figure(figsize=(5, 5))
_input1['Age'].plot.box(vert=False)
_input1 = _input1[_input1['Age'] <= 60]
plt.figure(figsize=(5, 5))
_input1['Age'].plot.box(vert=False)
expenses = ['FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
_input1.boxplot(column=expenses)
q_value = [i / 100 for i in range(95, 101, 1)]
q1 = _input1[expenses].quantile(q_value)
q1
_input1 = _input1[(_input1['FoodCourt'] <= 7993.12) & (_input1['ShoppingMall'] <= 2319.92) & (_input1['Spa'] <= 5255.6) & (_input1['VRDeck'] <= 5577.56) & (_input1['RoomService'] <= 3036.28)]
_input1.shape
_input1.boxplot(column=expenses)
_input1 = _input1[(_input1['FoodCourt'] <= 7993.12) & (_input1['ShoppingMall'] <= 2319.92) & (_input1['Spa'] <= 5255.6) & (_input1['VRDeck'] <= 5577.56) & (_input1['RoomService'] <= 3036.28)]
print(_input1.shape)
_input1.boxplot(column=expenses)
plt.figure(figsize=(7, 7))
plt.title('Distribution of Transported Passengers')
_input1['Transported'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.barplot(x='Transported', y='HomePlanet', data=_input1)
plt.title('Transported successfully from Home Planet')
plt.figure(figsize=(7, 7))
plt.title(' Passengers Confined to Cabins')
_input1['CryoSleep'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.barplot(x='Transported', y='CryoSleep', data=_input1)
plt.title('Confined Passengers Transported')
plt.figure(figsize=(7, 7))
_input1['Destination'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.barplot(x='Transported', y='Destination', data=_input1)
_input1['Side'] = _input1['Cabin'].str.split('/').str[2]
_input0['Side'] = _input0['Cabin'].str.split('/').str[2]
plt.figure(figsize=(7, 7))
_input1['Side'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.catplot(x='Side', y='Transported', kind='bar', palette='mako', data=_input1)
_input1['Deck'] = _input1['Cabin'].str.split('/').str[0]
_input0['Deck'] = _input0['Cabin'].str.split('/').str[0]
plt.figure(figsize=(7, 7))
_input1['Deck'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.catplot(x='Transported', y='Deck', kind='bar', palette='ch:.25', data=_input1)
plt.figure(figsize=(10, 10))
sns.catplot(x='HomePlanet', y='Transported', hue='Destination', kind='bar', palette='pastel', data=_input1)
plt.figure(figsize=(10, 10))
sns.catplot(x='HomePlanet', y='Age', hue='Transported', kind='box', palette='viridis', data=_input1)
plt.figure(figsize=(7, 7))
_input1['VIP'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(10, 10))
sns.catplot(x='Destination', y='VIP', hue='Transported', kind='point', palette='Spectral', data=_input1)
_input1['Expenses'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
plt.figure(figsize=(10, 10))
sns.catplot(x='VIP', y='Expenses', hue='Transported', kind='bar', palette='icefire', data=_input1)
plt.figure(figsize=(10, 10))
sns.catplot(x='HomePlanet', y='Expenses', hue='Transported', kind='bar', palette='coolwarm', data=_input1)
sns.scatterplot(x='Age', y='Expenses', data=_input1[_input1.Transported == True])
plt.figure(figsize=(7, 7))
sns.barplot(x='Side', y='Expenses', palette='ch:s=-.2,r=.6', data=_input1)
_input1.head()
_input0.head()
_input1 = _input1.drop(['PassengerId', 'Name', 'Expenses', 'Cabin'], inplace=False, axis=1)
_input0 = _input0.drop(['PassengerId', 'Name', 'Cabin'], inplace=False, axis=1)
_input1.head()
_input0.head()
categorical_var = [i for i in _input1.columns if _input1[i].dtypes == 'object']
for z in categorical_var:
    print(_input1[z].name, ':', _input1[z].unique())
numerical_var = _input1[[i for i in _input1.columns if _input1[i].dtypes != 'object']]
print(numerical_var)
numerical_var.plot(subplots=True, figsize=(8, 8))
categorical_var = [i for i in _input0.columns if _input0[i].dtypes == 'object']
for z in categorical_var:
    print(_input0[z].name, ':', _input0[z].unique())
numerical_var = _input0[[i for i in _input0.columns if _input0[i].dtypes != 'object']]
numerical_var
numerical_var.plot(subplots=True, figsize=(8, 8))
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for x in [i for i in _input1.columns if len(_input1[i].unique()) == 2]:
    _input1[x] = label_encoder.fit_transform(_input1[x])
_input1 = pd.get_dummies(_input1, columns=[i for i in _input1.columns if _input1[i].dtypes == 'object'], drop_first=True)
label_encoder = LabelEncoder()
for x in [i for i in _input0.columns if len(_input0[i].unique()) == 2]:
    _input0[x] = label_encoder.fit_transform(_input0[x])
_input0 = pd.get_dummies(_input0, columns=[i for i in _input0.columns if _input0[i].dtypes == 'object'], drop_first=True)
_input1.shape
_input0.shape
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
x_train = _input1.drop(['Transported'], axis=1)
x_test = _input0
Y_train = _input1['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(x_train, Y_train, test_size=0.25, random_state=0)
print('shape of X_train:', X_train.shape)
print('shape of y_train:', y_train.shape[0])
_input1.head()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = pd.DataFrame(X_train, columns=x_train.columns)
X_test = pd.DataFrame(X_test, columns=x_test.columns)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
seed = 7
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RFC', RandomForestClassifier(max_depth=20, random_state=200)))
models.append
results = []
names = []
scoring = 'accuracy'
for (name, model) in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
model = RandomForestClassifier(max_depth=100, random_state=42)