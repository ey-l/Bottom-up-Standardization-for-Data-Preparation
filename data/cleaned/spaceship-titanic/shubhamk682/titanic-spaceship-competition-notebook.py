import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
dataset = pd.read_csv('data/input/spaceship-titanic/train.csv')
dataset
dataset.isnull().sum()
data = dataset.iloc[:, [2, 3, 5, 6, 7, 10, 11, 13]]
data
data['VIP'].value_counts()
data.info()
data.isnull().sum()
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
data['CryoSleep'] = l.fit_transform(data['CryoSleep'])
data['Cabin'] = l.fit_transform(data['Cabin'])
data['VIP'] = l.fit_transform(data['VIP'])
data
data.info()
data.isnull().sum()
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['RoomService'].fillna(data['RoomService'].mean(), inplace=True)
data['Spa'].fillna(data['Spa'].mean(), inplace=True)
data['VRDeck'].fillna(data['VRDeck'].mean(), inplace=True)
data.isnull().sum()
sns.kdeplot(data['CryoSleep'])
for col in data.iloc[:, :-1].columns:
    if data.iloc[:, :-1][col].dtypes != 'object':
        (q1, q3) = (data.iloc[:, :-1][col].quantile(0.25), data.iloc[:, :-1][col].quantile(0.75))
        iqr = q3 - q1
        ll = q1 - 1.5 * iqr
        ul = q3 + 1.5 * iqr
        data['Age'] = np.where(data['Age'] > ul, data['Age'].mean(), np.where(data['Age'] < ll, data['Age'].mean(), data['Age']))
        data['VIP'] = np.where(data['VIP'] > ul, data['VIP'].mean(), np.where(data['VIP'] < ll, data['VIP'].mean(), data['VIP']))
        data['RoomService'] = np.where(data['RoomService'] > ul, data['RoomService'].mean(), np.where(data['RoomService'] < ll, data['RoomService'].mean(), data['RoomService']))
        data['Cabin'] = np.where(data['Cabin'] > ul, data['Cabin'].mean(), np.where(data['Cabin'] < ll, data['Cabin'].mean(), data['Cabin']))
        data['Spa'] = np.where(data['Spa'] > ul, data['Spa'].mean(), np.where(data['Spa'] < ll, data['Spa'].mean(), data['Spa']))
        data['VRDeck'] = np.where(data['VRDeck'] > ul, data['VRDeck'].mean(), np.where(data['VRDeck'] < ll, data['VRDeck'].mean(), data['VRDeck']))
Q1 = data.iloc[:, :-1].quantile(0.25)
Q3 = data.iloc[:, :-1].quantile(0.75)
IQR = Q3 - Q1
IQR
print('outlier Counter of the all features')
((data.iloc[:, :-1] < Q1 - 1.5 * IQR) | (data.iloc[:, :-1] > Q3 + 1.5 * IQR)).sum()
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
data['Transported'].value_counts()
from imblearn.under_sampling import NearMiss
nm = NearMiss()
(x_data, y_data) = nm.fit_resample(x, y)
from collections import Counter
print(Counter(y_data))
from sklearn.preprocessing import StandardScaler
ssd = StandardScaler()
x_scaling = ssd.fit_transform(x_data)
x_scaling
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x_scaling, y_data, test_size=0.2, random_state=20)
from sklearn.linear_model import LogisticRegression
logg = LogisticRegression()