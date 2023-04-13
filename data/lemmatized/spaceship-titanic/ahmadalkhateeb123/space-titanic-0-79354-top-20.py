import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import learning_curve
import numpy as np
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input1['HomePlanet'].value_counts().plot.bar()
plt.xlabel('Planets')
plt.ylabel('values')
_input1['CryoSleep'].value_counts().plot.bar()
plt.xlabel('CryoSleep')
plt.ylabel('values')
_input1['Destination'].value_counts().plot.bar(color=['red', 'green', 'blue'])
plt.xlabel('Destination')
plt.ylabel('values')
_input1['VIP'].value_counts().plot.bar()
plt.xlabel('VIP')
plt.ylabel('values')
_input1['Transported'].value_counts().plot.bar(color=['red', 'green'])
plt.xlabel('Transported')
plt.ylabel('values')
_input1['Transported'].value_counts().plot.pie()
plt.gca().set_aspect('equal')
sns.barplot(x='HomePlanet', y='Transported', hue='HomePlanet', data=_input1, palette='Blues')
_input1.isna().any()
_input1.isna().sum().sum() / _input1.shape[0] * 100
imp = SimpleImputer(strategy='mean')
_input1[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = pd.DataFrame(imp.fit_transform(_input1[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]))
imp_m = SimpleImputer(strategy='most_frequent')
_input1[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']] = pd.DataFrame(imp_m.fit_transform(_input1[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']]))
_input1.isna().any()
X = _input1.drop(columns=['PassengerId', 'Name', 'Destination', 'Transported', 'Cabin'], axis=1)
y = _input1['Transported']
X.head()
from sklearn.preprocessing import StandardScaler
scaler_s = StandardScaler()
X = pd.get_dummies(X, columns=['HomePlanet', 'CryoSleep', 'VIP'])
X[['RoomService', 'Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = pd.DataFrame(scaler_s.fit_transform(X[['RoomService', 'Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]))
X.head()
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, train_size=0.8)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=150, criterion='gini', max_depth=90, max_leaf_nodes=120, verbose=1)