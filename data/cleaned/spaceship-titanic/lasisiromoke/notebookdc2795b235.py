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
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.info()
print()
df_test.info()
df_train.head()
df_test.head()
print(f"The number of unique passengers in train data is {df_train['PassengerId'].nunique()}")
print()
print(f"The number of unique passengers in test data is {df_test['PassengerId'].nunique()}")
df_train = df_train.set_index('PassengerId')
df_train.head()
df_test = df_test.set_index('PassengerId')
df_test.head()
df_train.select_dtypes('number').describe()
df_test.select_dtypes('number').describe()
df_train['Age'] = np.where(df_train['Age'] == 0, np.nan, df_train['Age'])
df_test['Age'] = np.where(df_test['Age'] == 0, np.nan, df_test['Age'])
df_train.select_dtypes('number').describe()
df_test.select_dtypes('number').describe()
print(f"The number of unique Planet in train data is {df_train['HomePlanet'].nunique()}")
print()
print(f"The number of unique Planet is test data is {df_test['HomePlanet'].nunique()}")
print(df_train['HomePlanet'].value_counts().head())
print()
print(df_test['HomePlanet'].value_counts().head())
print(df_train['CryoSleep'].unique())
print()
print(df_test['CryoSleep'].unique())
print(df_train['CryoSleep'].value_counts().head())
print()
print(df_test['CryoSleep'].value_counts().head())
print(df_train['Destination'].value_counts())
print()
print(df_test['Destination'].value_counts())
Cabin = df_train['Cabin'].str.split('/', expand=True)
Cabin.nunique()
cabin = df_test['Cabin'].str.split('/', expand=True)
cabin.nunique()
df_train[['deck', 'num', 'side']] = Cabin.iloc[:, 0:3:1]
df_train.head()
df_test[['deck', 'num', 'side']] = cabin.iloc[:, 0:3:1]
df_test.head()
df_train.isnull().sum() / len(df_train) * 100
df_test.isnull().sum() / len(df_train) * 100
train_num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
train_cat_cols = ['HomePlanet', 'deck', 'num', 'side', 'CryoSleep', 'side', 'Destination', 'VIP', 'deck']
for i in train_num_cols:
    df_train[i] = df_train[i].fillna(df_train[i].median())
for i in train_cat_cols:
    df_train[i] = df_train[i].fillna(st.mode(df_train[i]))
test_num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
test_cat_cols = ['HomePlanet', 'deck', 'num', 'side', 'CryoSleep', 'side', 'Destination', 'VIP', 'deck']
for i in test_num_cols:
    df_test[i] = df_test[i].fillna(df_test[i].median())
for i in test_cat_cols:
    df_test[i] = df_test[i].fillna(st.mode(df_train[i]))
df_train.select_dtypes('object').nunique()
df_train.select_dtypes('object').nunique().plot.bar(figsize=(12, 5))
plt.ylabel('Number of unique Categories')
plt.xlabel('Variable')
plt.title('Cardianlity')
df_test.select_dtypes('object').nunique()
df_train.select_dtypes('object').nunique().plot.bar(figsize=(12, 5))
plt.ylabel('Number of unique Categories')
plt.xlabel('Variable')
plt.title('Cardianlity')
df_train.drop(columns=['Cabin', 'Name', 'num'], inplace=True)
df_test.drop(columns=['Cabin', 'Name', 'num'], inplace=True)
df_train['Transported'].value_counts(normalize=True).plot(kind='bar', xlabel='Transported', ylabel='Frequency', title='Class Balance')
df_train['Transported'].value_counts()
df_train['ShoppingMall'].hist()
plt.xlabel('Shopping bill')
(plt.ylabel('Count'),)
plt.title('Distribution of amount spent in Shopping Mall')
sns.boxplot(x='Transported', y='ShoppingMall', data=df_train)
plt.xlabel('Transported')
(q1, q9) = df_train['ShoppingMall'].quantile([0.01, 0.99])
mask = df_train['ShoppingMall'].between(q1, q9)
sns.boxplot(x='Transported', y='ShoppingMall', data=df_train[mask])
corr = df_train.drop(columns='Transported').corr()
sns.heatmap(corr, annot=True)
corr = df_test.corr()
sns.heatmap(corr, annot=True)
target = 'Transported'
X = df_train.drop(columns='Transported')
y = df_train[target]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)
acc_baseline = y_train.value_counts(normalize=True).max()
print('Baseline Accuracy:', round(acc_baseline, 4))
model = make_pipeline(OneHotEncoder(use_cat_names=True), LogisticRegression(max_iter=1000))