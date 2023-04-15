import pandas as pd
import numpy as np
import re
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.shape
test.shape
train.head()
test.head()
train.groupby('HomePlanet')['RoomService'].describe()
train.groupby('HomePlanet')['Age'].describe()
train.isnull().sum()
test.isnull().sum()
train['HomePlanet'].value_counts().plot.bar()
train.info()
train.corr()
train.describe(include='object')
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(train.corr(), annot=True)
cat_col = ['HomePlanet', 'CryoSleep', 'VIP']
sns.set(style='ticks')
for i in cat_col:
    sns.catplot(x=i, y='Transported', data=train, kind='point', aspect=2)
    plt.ylim(0, 1)
    plt.grid()

sns.kdeplot(x='Age', data=train, hue='Transported')
plt.grid()

features_na = [features for features in train.columns if train[features].isnull().sum() > 1]
for features in features_na:
    print(features, np.round(train[features].isnull().sum() > 1), '% missing values')
for features in features_na:
    data = train.copy()
    data[features] = np.where(data[features].isnull(), 1, 0)
    data.groupby(features)['Transported'].median().plot.bar()
    plt.title('features')


train.head()
y_train_1 = train['Transported']
y_train_int = np.array(y_train_1)
y_train_array = y_train_int.astype(int)
y_train_1 = pd.DataFrame(y_train_array)
y_train_list = [int(item) for item in y_train_1]
train.head()
train = train.drop(['Transported', 'Name', 'PassengerId'], axis=1)
test = test.drop(['Name', 'PassengerId'], axis=1)
train.head()
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
category = ce.OrdinalEncoder(cols=['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP'], return_df=True)
data_2 = category.fit_transform(train)
test_data = category.fit_transform(test)
data_2.head()
data_2.isnull().sum()
import missingno as msno
msno.matrix(data_2)
from sklearn.impute import SimpleImputer
imputing_values = SimpleImputer(strategy='mean')
data_simple = imputing_values.fit_transform(data_2)
test = imputing_values.fit_transform(test_data)
pd.DataFrame(test).isnull().sum()
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