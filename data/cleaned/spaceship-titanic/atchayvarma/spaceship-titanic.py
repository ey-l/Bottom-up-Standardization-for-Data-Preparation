import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder as le
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from scipy.stats import randint
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
tr_path = 'data/input/spaceship-titanic/train.csv'
ts_path = 'data/input/spaceship-titanic/test.csv'
tr_df = pd.read_csv(tr_path)
tr_df.head()
ts_df = pd.read_csv(ts_path)
ts_df.head()
print(f'Training Dataset(Row,Columns): {tr_df.shape}')
print(f'Test Dataset(Row,Columns): {ts_df.shape}')
tr_df.info()
tr_df.describe()
tr_df[['CabinDeck', 'CabinNo', 'CabinSide']] = tr_df['Cabin'].str.split('/', expand=True)
ts_df[['CabinDeck', 'CabinNo', 'CabinSide']] = ts_df['Cabin'].str.split('/', expand=True)
tr_df.drop(['Name', 'Cabin'], axis=1, inplace=True)
ts_df.drop(['Name', 'Cabin'], axis=1, inplace=True)
tr_df.isnull().sum().sort_values(ascending=False)
null_cols = ['CryoSleep', 'ShoppingMall', 'VIP', 'HomePlanet', 'VRDeck', 'FoodCourt', 'Spa', 'Destination', 'RoomService', 'Age', 'CabinDeck', 'CabinNo', 'CabinSide']
for col in null_cols:
    print(f'{col}:\n{tr_df[col].value_counts()}\n', '-' * 50)
    tr_df[col] = tr_df[col].fillna(tr_df[col].dropna().mode().values[0])
print(f'After filling NA Values\n', '#' * 50)
tr_df.isnull().sum().sort_values(ascending=False)
num = tr_df.select_dtypes('number').columns.to_list()
cat = tr_df.select_dtypes(['object', 'bool']).columns.to_list()
tr_num_df = tr_df[num]
tr_cat_df = tr_df[cat]
print(tr_df['Transported'].value_counts())
sns.set(style='whitegrid')
sns.countplot(x=tr_df['Transported'])

for i in tr_num_df:
    plt.hist(tr_num_df[i])
    plt.title(i)

for i in cat[1:5]:
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    sns.countplot(x=i, data=tr_df, hue='Transported')
    plt.xlabel(i, fontsize=14)
tr_df.dtypes
tr_df = pd.get_dummies(tr_df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'CabinDeck', 'CabinSide'], drop_first=True)
ts_df = pd.get_dummies(ts_df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'CabinDeck', 'CabinSide'], drop_first=True)
print(tr_df.info(), '\n\n', ts_df.info())
sns.heatmap(tr_df.corr(), cmap='cubehelix_r')
tr_df.corr().style.background_gradient(cmap='coolwarm')
y = tr_df['Transported']
X = tr_df.drop('Transported', axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)
DT = DecisionTreeClassifier(random_state=6)