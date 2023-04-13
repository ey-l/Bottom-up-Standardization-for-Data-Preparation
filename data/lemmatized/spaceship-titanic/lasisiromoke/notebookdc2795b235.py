import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from category_encoders import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
import statistics as st
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.info()
print()
_input0.info()
_input1.head()
_input0.head()
print(f"The number of unique passengers in train data is {_input1['PassengerId'].nunique()}")
print()
print(f"The number of unique passengers in test data is {_input0['PassengerId'].nunique()}")
_input1 = _input1.set_index('PassengerId')
_input1.head()
_input0 = _input0.set_index('PassengerId')
_input0.head()
_input1.select_dtypes('number').describe()
_input0.select_dtypes('number').describe()
_input1['Age'] = np.where(_input1['Age'] == 0, np.nan, _input1['Age'])
_input0['Age'] = np.where(_input0['Age'] == 0, np.nan, _input0['Age'])
_input1.select_dtypes('number').describe()
_input0.select_dtypes('number').describe()
print(f"The number of unique Planet in train data is {_input1['HomePlanet'].nunique()}")
print()
print(f"The number of unique Planet is test data is {_input0['HomePlanet'].nunique()}")
print(_input1['HomePlanet'].value_counts().head())
print()
print(_input0['HomePlanet'].value_counts().head())
print(_input1['CryoSleep'].unique())
print()
print(_input0['CryoSleep'].unique())
print(_input1['CryoSleep'].value_counts().head())
print()
print(_input0['CryoSleep'].value_counts().head())
print(_input1['Destination'].value_counts())
print()
print(_input0['Destination'].value_counts())
Cabin = _input1['Cabin'].str.split('/', expand=True)
Cabin.nunique()
cabin = _input0['Cabin'].str.split('/', expand=True)
cabin.nunique()
_input1[['deck', 'num', 'side']] = Cabin.iloc[:, 0:3:1]
_input1.head()
_input0[['deck', 'num', 'side']] = cabin.iloc[:, 0:3:1]
_input0.head()
_input1.isnull().sum() / len(_input1) * 100
_input0.isnull().sum() / len(_input1) * 100
train_num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
train_cat_cols = ['HomePlanet', 'deck', 'num', 'side', 'CryoSleep', 'side', 'Destination', 'VIP', 'deck']
for i in train_num_cols:
    _input1[i] = _input1[i].fillna(_input1[i].median())
for i in train_cat_cols:
    _input1[i] = _input1[i].fillna(st.mode(_input1[i]))
test_num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
test_cat_cols = ['HomePlanet', 'deck', 'num', 'side', 'CryoSleep', 'side', 'Destination', 'VIP', 'deck']
for i in test_num_cols:
    _input0[i] = _input0[i].fillna(_input0[i].median())
for i in test_cat_cols:
    _input0[i] = _input0[i].fillna(st.mode(_input1[i]))
_input1.select_dtypes('object').nunique()
_input1.select_dtypes('object').nunique().plot.bar(figsize=(12, 5))
plt.ylabel('Number of unique Categories')
plt.xlabel('Variable')
plt.title('Cardianlity')
_input0.select_dtypes('object').nunique()
_input1.select_dtypes('object').nunique().plot.bar(figsize=(12, 5))
plt.ylabel('Number of unique Categories')
plt.xlabel('Variable')
plt.title('Cardianlity')
_input1 = _input1.drop(columns=['Cabin', 'Name', 'num'], inplace=False)
_input0 = _input0.drop(columns=['Cabin', 'Name', 'num'], inplace=False)
_input1['Transported'].value_counts(normalize=True).plot(kind='bar', xlabel='Transported', ylabel='Frequency', title='Class Balance')
_input1['Transported'].value_counts()
_input1['ShoppingMall'].hist()
plt.xlabel('Shopping bill')
(plt.ylabel('Count'),)
plt.title('Distribution of amount spent in Shopping Mall')
sns.boxplot(x='Transported', y='ShoppingMall', data=_input1)
plt.xlabel('Transported')
(q1, q9) = _input1['ShoppingMall'].quantile([0.01, 0.99])
mask = _input1['ShoppingMall'].between(q1, q9)
sns.boxplot(x='Transported', y='ShoppingMall', data=_input1[mask])
corr = _input1.drop(columns='Transported').corr()
sns.heatmap(corr, annot=True)
corr = _input0.corr()
sns.heatmap(corr, annot=True)
target = 'Transported'
X = _input1.drop(columns='Transported')
y = _input1[target]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)
acc_baseline = y_train.value_counts(normalize=True).max()
print('Baseline Accuracy:', round(acc_baseline, 4))
model = make_pipeline(OneHotEncoder(use_cat_names=True), LogisticRegression(max_iter=1000))