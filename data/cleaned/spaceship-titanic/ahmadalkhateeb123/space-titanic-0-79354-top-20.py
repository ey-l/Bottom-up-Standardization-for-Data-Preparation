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
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df
df['HomePlanet'].value_counts().plot.bar()
plt.xlabel('Planets')
plt.ylabel('values')
df['CryoSleep'].value_counts().plot.bar()
plt.xlabel('CryoSleep')
plt.ylabel('values')
df['Destination'].value_counts().plot.bar(color=['red', 'green', 'blue'])
plt.xlabel('Destination')
plt.ylabel('values')
df['VIP'].value_counts().plot.bar()
plt.xlabel('VIP')
plt.ylabel('values')
df['Transported'].value_counts().plot.bar(color=['red', 'green'])
plt.xlabel('Transported')
plt.ylabel('values')
df['Transported'].value_counts().plot.pie()
plt.gca().set_aspect('equal')
sns.barplot(x='HomePlanet', y='Transported', hue='HomePlanet', data=df, palette='Blues')
df.isna().any()
df.isna().sum().sum() / df.shape[0] * 100
imp = SimpleImputer(strategy='mean')
df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = pd.DataFrame(imp.fit_transform(df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]))
imp_m = SimpleImputer(strategy='most_frequent')
df[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']] = pd.DataFrame(imp_m.fit_transform(df[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']]))
df.isna().any()
X = df.drop(columns=['PassengerId', 'Name', 'Destination', 'Transported', 'Cabin'], axis=1)
y = df['Transported']
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