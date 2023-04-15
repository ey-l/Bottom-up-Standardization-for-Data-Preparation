import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train
sample_submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
sample_submission
train.head(10)
train.info()
del train['PassengerId']
del train['Cabin']
del train['Destination']
del train['ShoppingMall']
del train['Name']
del train['HomePlanet']
train.describe()
from sklearn.preprocessing import LabelEncoder
L = LabelEncoder()
train['CryoSleep'] = L.fit_transform(train['CryoSleep'])
train['VIP'] = L.fit_transform(train['VIP'])
train['Transported'] = L.fit_transform(train['Transported'])
train
train.isnull().sum()
train['Age'].mean()
train['RoomService'].mean()
train['FoodCourt'].mean()
train['Spa'].mean()
train['VRDeck'].mean()
train['Age'] = train['Age'].fillna(28)
train['RoomService'] = train['RoomService'].fillna(224)
train['FoodCourt'] = train['FoodCourt'].fillna(458)
train['Spa'] = train['Spa'].fillna(311)
train['VRDeck'] = train['VRDeck'].fillna(304)
train
Q1 = train.quantile(0.25)
Q3 = train.quantile(0.75)
IQR = Q3 - Q1
IQR
print('outlier Counter of the all features')
((train < Q1 - 1.5 * IQR) | (train > Q3 + 1.5 * IQR)).sum()
for col in train.columns:
    if train[col].dtypes != 'object':
        (q1, q3) = (train[col].quantile(0.25), train[col].quantile(0.75))
        iqr = q3 - q1
        ll = q1 - 1.5 * iqr
        ul = q3 + 1.5 * iqr
        train['Age'] = np.where(train['Age'] > ul, train['Age'].mean(), np.where(train['Age'] < ll, train['Age'].mean(), train['Age']))
        train['VIP'] = np.where(train['VIP'] > ul, train['VIP'].mean(), np.where(train['VIP'] < ll, train['VIP'].mean(), train['VIP']))
        train['RoomService'] = np.where(train['RoomService'] > ul, train['RoomService'].mean(), np.where(train['RoomService'] < ll, train['RoomService'].mean(), train['RoomService']))
        train['FoodCourt'] = np.where(train['FoodCourt'] > ul, train['FoodCourt'].mean(), np.where(train['FoodCourt'] < ll, train['FoodCourt'].mean(), train['Age']))
        train['Spa'] = np.where(train['Spa'] > ul, train['Spa'].mean(), np.where(train['Spa'] < ll, train['Spa'].mean(), train['Spa']))
        train['VRDeck'] = np.where(train['VRDeck'] > ul, train['VRDeck'].mean(), np.where(train['VRDeck'] < ll, train['VRDeck'].mean(), train['VRDeck']))
Q1 = train.quantile(0.25)
Q3 = train.quantile(0.75)
IQR = Q3 - Q1
IQR
print('outlier Counter of the all features')
((train < Q1 - 1.5 * IQR) | (train > Q3 + 1.5 * IQR)).sum()
train.isnull().sum()
train['Transported'].value_counts()
x = train.iloc[:, :-1].values
y = train.iloc[:, -1].values
y
from sklearn.preprocessing import StandardScaler
S = StandardScaler()
X = S.fit_transform(x)
X
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=25)
from collections import Counter
Counter(y_train)
from sklearn.ensemble import RandomForestClassifier
R = RandomForestClassifier()