import pandas as pd
import numpy as np
import re
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.shape
_input0.shape
_input1.head()
_input0.head()
_input1.groupby('HomePlanet')['RoomService'].describe()
_input1.groupby('HomePlanet')['Age'].describe()
_input1.isnull().sum()
_input0.isnull().sum()
_input1['HomePlanet'].value_counts().plot.bar()
_input1.info()
_input1.corr()
_input1.describe(include='object')
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(_input1.corr(), annot=True)
cat_col = ['HomePlanet', 'CryoSleep', 'VIP']
sns.set(style='ticks')
for i in cat_col:
    sns.catplot(x=i, y='Transported', data=_input1, kind='point', aspect=2)
    plt.ylim(0, 1)
    plt.grid()
sns.kdeplot(x='Age', data=_input1, hue='Transported')
plt.grid()
features_na = [features for features in _input1.columns if _input1[features].isnull().sum() > 1]
for features in features_na:
    print(features, np.round(_input1[features].isnull().sum() > 1), '% missing values')
for features in features_na:
    data = _input1.copy()
    data[features] = np.where(data[features].isnull(), 1, 0)
    data.groupby(features)['Transported'].median().plot.bar()
    plt.title('features')
_input1.head()
y_train_1 = _input1['Transported']
y_train_int = np.array(y_train_1)
y_train_array = y_train_int.astype(int)
y_train_1 = pd.DataFrame(y_train_array)
y_train_list = [int(item) for item in y_train_1]
_input1.head()
_input1 = _input1.drop(['Transported', 'Name', 'PassengerId'], axis=1)
_input0 = _input0.drop(['Name', 'PassengerId'], axis=1)
_input1.head()
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
category = ce.OrdinalEncoder(cols=['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP'], return_df=True)
data_2 = category.fit_transform(_input1)
test_data = category.fit_transform(_input0)
data_2.head()
data_2.isnull().sum()
import missingno as msno
msno.matrix(data_2)
from sklearn.impute import SimpleImputer
imputing_values = SimpleImputer(strategy='mean')
data_simple = imputing_values.fit_transform(data_2)
_input0 = imputing_values.fit_transform(test_data)
pd.DataFrame(_input0).isnull().sum()
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
knn_data = imputer.fit_transform(data_2)
knn_data = pd.DataFrame(knn_data)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(knn_data)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
X_train_scaled = pd.DataFrame(scaled)
len(X_train_scaled)
len(y_train_1)
y_train_1.values.ravel()
logreg = LogisticRegression()