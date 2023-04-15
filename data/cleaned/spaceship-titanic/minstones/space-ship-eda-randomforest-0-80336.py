import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.head()
df_train.info()
df_train.isna().sum()
df_test.isna().sum()
from sklearn.impute import SimpleImputer
train_columns = df_train.columns
test_columns = df_test.columns
imputer = SimpleImputer(strategy='most_frequent')
df_train = imputer.fit_transform(df_train)
df_test = imputer.fit_transform(df_test)
print(df_train)
df_train = pd.DataFrame(df_train, columns=train_columns)
df_test = pd.DataFrame(df_test, columns=test_columns)
df_train.head(5)
cabin_columns = ['Deck', 'Deck Number', 'Side']
id_columns = ['Passenger Group', 'Passenger Number']
sepr_cabin = df_train['Cabin'].str.split('/', n=-1, expand=True)
sepr_id = df_train['PassengerId'].str.split('_', n=-1, expand=True)
sepr_cabin.columns = cabin_columns
sepr_id.columns = id_columns
sepr_cabin_test = df_test['Cabin'].str.split('/', n=-1, expand=True)
sepr_id_test = df_test['PassengerId'].str.split('_', n=-1, expand=True)
sepr_cabin_test.columns = cabin_columns
sepr_id_test.columns = id_columns
df_train = pd.concat([df_train, sepr_cabin, sepr_id], axis=1)
df_test = pd.concat([df_test, sepr_cabin_test, sepr_id_test], axis=1)
df_train.drop(columns=['PassengerId', 'Cabin'], inplace=True)
df_test.drop(columns=['PassengerId', 'Cabin'], inplace=True)
df_train.head(5)
df_train.describe(include=['O'])
sns.catplot(x='HomePlanet', y='Transported', kind='bar', data=df_train)
sns.catplot(x='CryoSleep', y='Transported', kind='bar', data=df_train)
sns.catplot(x='Destination', y='Transported', kind='bar', data=df_train)
sns.catplot(x='VIP', y='Transported', kind='bar', data=df_train)
sns.catplot(x='Deck', y='Transported', kind='bar', data=df_train)
sns.catplot(x='Side', y='Transported', kind='bar', data=df_train)
sns.catplot(x='Passenger Number', y='Transported', kind='bar', data=df_train)
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
cat_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Passenger Number']
num_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
encoder_train = OrdinalEncoder().fit_transform(df_train[cat_columns])
encoder_train = pd.DataFrame(encoder_train, columns=cat_columns)
encoder_test = OrdinalEncoder().fit_transform(df_test[cat_columns])
encoder_test = pd.DataFrame(encoder_test, columns=cat_columns)
x_train = pd.concat([encoder_train, df_train[num_columns]], axis=1)
x_test = pd.concat([encoder_test, df_test[num_columns]], axis=1)
y_train = df_train['Transported']
label_encoder = LabelEncoder().fit_transform(y_train)
y_train = pd.DataFrame(label_encoder, columns=['Transported'])
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
model_rf = RandomForestClassifier(max_depth=10, random_state=42)