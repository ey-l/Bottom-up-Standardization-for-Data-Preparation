import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.info()
_input1.isna().sum()
_input0.isna().sum()
from sklearn.impute import SimpleImputer
train_columns = _input1.columns
test_columns = _input0.columns
imputer = SimpleImputer(strategy='most_frequent')
_input1 = imputer.fit_transform(_input1)
_input0 = imputer.fit_transform(_input0)
print(_input1)
_input1 = pd.DataFrame(_input1, columns=train_columns)
_input0 = pd.DataFrame(_input0, columns=test_columns)
_input1.head(5)
cabin_columns = ['Deck', 'Deck Number', 'Side']
id_columns = ['Passenger Group', 'Passenger Number']
sepr_cabin = _input1['Cabin'].str.split('/', n=-1, expand=True)
sepr_id = _input1['PassengerId'].str.split('_', n=-1, expand=True)
sepr_cabin.columns = cabin_columns
sepr_id.columns = id_columns
sepr_cabin_test = _input0['Cabin'].str.split('/', n=-1, expand=True)
sepr_id_test = _input0['PassengerId'].str.split('_', n=-1, expand=True)
sepr_cabin_test.columns = cabin_columns
sepr_id_test.columns = id_columns
_input1 = pd.concat([_input1, sepr_cabin, sepr_id], axis=1)
_input0 = pd.concat([_input0, sepr_cabin_test, sepr_id_test], axis=1)
_input1 = _input1.drop(columns=['PassengerId', 'Cabin'], inplace=False)
_input0 = _input0.drop(columns=['PassengerId', 'Cabin'], inplace=False)
_input1.head(5)
_input1.describe(include=['O'])
sns.catplot(x='HomePlanet', y='Transported', kind='bar', data=_input1)
sns.catplot(x='CryoSleep', y='Transported', kind='bar', data=_input1)
sns.catplot(x='Destination', y='Transported', kind='bar', data=_input1)
sns.catplot(x='VIP', y='Transported', kind='bar', data=_input1)
sns.catplot(x='Deck', y='Transported', kind='bar', data=_input1)
sns.catplot(x='Side', y='Transported', kind='bar', data=_input1)
sns.catplot(x='Passenger Number', y='Transported', kind='bar', data=_input1)
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
cat_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Passenger Number']
num_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
encoder_train = OrdinalEncoder().fit_transform(_input1[cat_columns])
encoder_train = pd.DataFrame(encoder_train, columns=cat_columns)
encoder_test = OrdinalEncoder().fit_transform(_input0[cat_columns])
encoder_test = pd.DataFrame(encoder_test, columns=cat_columns)
x_train = pd.concat([encoder_train, _input1[num_columns]], axis=1)
x_test = pd.concat([encoder_test, _input0[num_columns]], axis=1)
y_train = _input1['Transported']
label_encoder = LabelEncoder().fit_transform(y_train)
y_train = pd.DataFrame(label_encoder, columns=['Transported'])
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
model_rf = RandomForestClassifier(max_depth=10, random_state=42)