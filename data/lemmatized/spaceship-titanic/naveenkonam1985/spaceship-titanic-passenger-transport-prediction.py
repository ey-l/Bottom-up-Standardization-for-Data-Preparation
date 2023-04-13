import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.sample(5)

def split_columns(df, delim):
    return df.str.split(delim, n=-1, expand=True)
cabin_columns = ['Deck', 'Deck Number', 'Side']
id_columns = ['Passenger Group', 'Passenger Number']
split_columns_cabin_train = split_columns(_input1['Cabin'], '/')
split_columns_id_train = split_columns(_input1['PassengerId'], '_')
split_columns_cabin_train.columns = cabin_columns
split_columns_id_train.columns = id_columns
split_columns_cabin_test = split_columns(_input0['Cabin'], '/')
split_columns_id_test = split_columns(_input0['PassengerId'], '_')
split_columns_cabin_test.columns = cabin_columns
split_columns_id_test.columns = id_columns
_input1 = _input1.drop(['PassengerId', 'Cabin'], axis=1)
_input0 = _input0.drop(['PassengerId', 'Cabin'], axis=1)
_input1 = pd.concat([_input1, split_columns_cabin_train, split_columns_id_train], axis=1)
_input0 = pd.concat([_input0, split_columns_cabin_test, split_columns_id_test], axis=1)
_input1.isna().sum()
_input0.isna().sum()
from sklearn.impute import SimpleImputer
columns_train = _input1.columns
columns_test = _input0.columns
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
_input1 = imputer.fit_transform(_input1)
_input0 = imputer.fit_transform(_input0)
_input1 = pd.DataFrame(_input1, columns=columns_train)
_input0 = pd.DataFrame(_input0, columns=columns_test)
import matplotlib.pyplot as plt
import seaborn as sns
_input1.info()
plt.figure(figsize=(10, 8))
sns.histplot(data=_input1, x='Age', kde=True)
plt.title('Histogram of Age')
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=_input1, x='HomePlanet')
ax.bar_label(ax.containers[0])
plt.title('Number of Travellers based on Home Planet')
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=_input1, x='Destination')
ax.bar_label(ax.containers[0])
plt.title('Number of Travellers based on Destination')
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=_input1, x='VIP')
ax.bar_label(ax.containers[0])
plt.title('VIP Count')
plt.figure(figsize=(12, 8))
plt.subplot(122)
ax = sns.countplot(data=_input1, x='Transported', hue='HomePlanet')
for i in range(len(ax.containers)):
    ax.bar_label(ax.containers[i])
travelled_count = _input1['Transported'].value_counts()
plt.subplot(121)
plt.pie(travelled_count, autopct='%.2f', labels=travelled_count.index)
plt.suptitle('Number of Tarevellers Transported')
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=_input1, x='Deck', hue='Transported')
for i in range(len(ax.containers)):
    ax.bar_label(ax.containers[i])
plt.title('Deck')
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
num_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck Number', 'Passenger Group', 'Passenger Number']
cat_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side']
for col in num_columns:
    _input1[col] = pd.to_numeric(_input1[col])
encoder_train = OrdinalEncoder().fit_transform(_input1[cat_columns])
encoder_train = pd.DataFrame(encoder_train, columns=cat_columns)
encoder_test = OrdinalEncoder().fit_transform(_input0[cat_columns])
encoder_test = pd.DataFrame(encoder_test, columns=cat_columns)
X_tr = pd.concat([_input1[num_columns], encoder_train], axis=1)
X_test = pd.concat([_input0[num_columns], encoder_test], axis=1)
y_tr = _input1['Transported']
label_encoder = LabelEncoder().fit_transform(y_tr)
y_tr = pd.DataFrame(label_encoder, columns=['Transported'])
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from warnings import filterwarnings
filterwarnings('ignore')
(X_train, X_valid, y_train, y_valid) = train_test_split(X_tr, y_tr, test_size=0.25, random_state=42)
model_dt = DecisionTreeClassifier(max_depth=10)