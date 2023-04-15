import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
space_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
space_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
sample = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
space_train.shape
space_test.shape
space_train.head()
space_test.head()
space_train.info()
space_train.describe()
space_train.PassengerId.nunique() / space_train.shape[0]
space_train.Transported.value_counts()
space_train.isnull().sum()
null_value = space_test.isnull().sum()
null_value
space_train['HomePlanet'].fillna(method='ffill', inplace=True)
space_train['CryoSleep'].fillna(method='ffill', inplace=True)
space_train['Cabin'].fillna(method='ffill', inplace=True)
space_train['Destination'].fillna(method='ffill', inplace=True)
space_train['Age'].fillna(space_train['Age'].mean(), inplace=True)
space_train['VIP'].fillna(method='ffill', inplace=True)
space_train['RoomService'].fillna(space_train['RoomService'].mean(), inplace=True)
space_train['FoodCourt'].fillna(space_train['FoodCourt'].mean(), inplace=True)
space_train['ShoppingMall'].fillna(space_train['ShoppingMall'].mean(), inplace=True)
space_train['Spa'].fillna(space_train['Spa'].mean(), inplace=True)
space_train['VRDeck'].fillna(space_train['VRDeck'].mean(), inplace=True)
space_train['Name'].fillna(method='ffill', inplace=True)
space_train.isnull().sum()
space_test['HomePlanet'].fillna(method='ffill', inplace=True)
space_test['CryoSleep'].fillna(method='ffill', inplace=True)
space_test['Cabin'].fillna(method='ffill', inplace=True)
space_test['Destination'].fillna(method='ffill', inplace=True)
space_test['Age'].fillna(space_test['Age'].mean(), inplace=True)
space_test['VIP'].fillna(method='ffill', inplace=True)
space_test['RoomService'].fillna(space_test['RoomService'].mean(), inplace=True)
space_test['FoodCourt'].fillna(space_test['FoodCourt'].mean(), inplace=True)
space_test['ShoppingMall'].fillna(space_test['ShoppingMall'].mean(), inplace=True)
space_test['Spa'].fillna(space_test['Spa'].mean(), inplace=True)
space_test['VRDeck'].fillna(space_test['VRDeck'].mean(), inplace=True)
space_test['Name'].fillna(method='ffill', inplace=True)
space_test.isnull().sum()
plt.figure(figsize=(5, 5))
space_train['Age'].plot.box(vert=False)
space_train = space_train[space_train['Age'] <= 60]
plt.figure(figsize=(5, 5))
space_train['Age'].plot.box(vert=False)
expenses = ['FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
space_train.boxplot(column=expenses)
q_value = [i / 100 for i in range(95, 101, 1)]
q1 = space_train[expenses].quantile(q_value)
q1
space_train = space_train[(space_train['FoodCourt'] <= 7993.12) & (space_train['ShoppingMall'] <= 2319.92) & (space_train['Spa'] <= 5255.6) & (space_train['VRDeck'] <= 5577.56) & (space_train['RoomService'] <= 3036.28)]
space_train.shape
space_train.boxplot(column=expenses)
space_train = space_train[(space_train['FoodCourt'] <= 7993.12) & (space_train['ShoppingMall'] <= 2319.92) & (space_train['Spa'] <= 5255.6) & (space_train['VRDeck'] <= 5577.56) & (space_train['RoomService'] <= 3036.28)]
print(space_train.shape)
space_train.boxplot(column=expenses)
plt.figure(figsize=(7, 7))
plt.title('Distribution of Transported Passengers')
space_train['Transported'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.barplot(x='Transported', y='HomePlanet', data=space_train)
plt.title('Transported successfully from Home Planet')
plt.figure(figsize=(7, 7))
plt.title(' Passengers Confined to Cabins')
space_train['CryoSleep'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.barplot(x='Transported', y='CryoSleep', data=space_train)
plt.title('Confined Passengers Transported')
plt.figure(figsize=(7, 7))
space_train['Destination'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.barplot(x='Transported', y='Destination', data=space_train)
space_train['Side'] = space_train['Cabin'].str.split('/').str[2]
space_test['Side'] = space_test['Cabin'].str.split('/').str[2]
plt.figure(figsize=(7, 7))
space_train['Side'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.catplot(x='Side', y='Transported', kind='bar', palette='mako', data=space_train)
space_train['Deck'] = space_train['Cabin'].str.split('/').str[0]
space_test['Deck'] = space_test['Cabin'].str.split('/').str[0]
plt.figure(figsize=(7, 7))
space_train['Deck'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.catplot(x='Transported', y='Deck', kind='bar', palette='ch:.25', data=space_train)
plt.figure(figsize=(10, 10))
sns.catplot(x='HomePlanet', y='Transported', hue='Destination', kind='bar', palette='pastel', data=space_train)
plt.figure(figsize=(10, 10))
sns.catplot(x='HomePlanet', y='Age', hue='Transported', kind='box', palette='viridis', data=space_train)
plt.figure(figsize=(7, 7))
space_train['VIP'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(10, 10))
sns.catplot(x='Destination', y='VIP', hue='Transported', kind='point', palette='Spectral', data=space_train)
space_train['Expenses'] = space_train['RoomService'] + space_train['FoodCourt'] + space_train['ShoppingMall'] + space_train['Spa'] + space_train['VRDeck']
plt.figure(figsize=(10, 10))
sns.catplot(x='VIP', y='Expenses', hue='Transported', kind='bar', palette='icefire', data=space_train)
plt.figure(figsize=(10, 10))
sns.catplot(x='HomePlanet', y='Expenses', hue='Transported', kind='bar', palette='coolwarm', data=space_train)
sns.scatterplot(x='Age', y='Expenses', data=space_train[space_train.Transported == True])
plt.figure(figsize=(7, 7))
sns.barplot(x='Side', y='Expenses', palette='ch:s=-.2,r=.6', data=space_train)
space_train.head()
space_test.head()
space_train.drop(['PassengerId', 'Name', 'Expenses', 'Cabin'], inplace=True, axis=1)
space_test.drop(['PassengerId', 'Name', 'Cabin'], inplace=True, axis=1)
space_train.head()
space_test.head()
categorical_var = [i for i in space_train.columns if space_train[i].dtypes == 'object']
for z in categorical_var:
    print(space_train[z].name, ':', space_train[z].unique())
numerical_var = space_train[[i for i in space_train.columns if space_train[i].dtypes != 'object']]
print(numerical_var)
numerical_var.plot(subplots=True, figsize=(8, 8))
categorical_var = [i for i in space_test.columns if space_test[i].dtypes == 'object']
for z in categorical_var:
    print(space_test[z].name, ':', space_test[z].unique())
numerical_var = space_test[[i for i in space_test.columns if space_test[i].dtypes != 'object']]
numerical_var
numerical_var.plot(subplots=True, figsize=(8, 8))
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for x in [i for i in space_train.columns if len(space_train[i].unique()) == 2]:
    space_train[x] = label_encoder.fit_transform(space_train[x])
space_train = pd.get_dummies(space_train, columns=[i for i in space_train.columns if space_train[i].dtypes == 'object'], drop_first=True)
label_encoder = LabelEncoder()
for x in [i for i in space_test.columns if len(space_test[i].unique()) == 2]:
    space_test[x] = label_encoder.fit_transform(space_test[x])
space_test = pd.get_dummies(space_test, columns=[i for i in space_test.columns if space_test[i].dtypes == 'object'], drop_first=True)
space_train.shape
space_test.shape
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
x_train = space_train.drop(['Transported'], axis=1)
x_test = space_test
Y_train = space_train['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(x_train, Y_train, test_size=0.25, random_state=0)
print('shape of X_train:', X_train.shape)
print('shape of y_train:', y_train.shape[0])
space_train.head()
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