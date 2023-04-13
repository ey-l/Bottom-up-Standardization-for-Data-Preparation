import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pandas_profiling import ProfileReport
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
ProfileReport(_input1, title='EDA Report Spaceship Titanic')
round(_input1.isna().sum() / len(_input1) * 100, 2)
plt.figure(figsize=(10, 6))
sns.displot(data=_input1.isna().melt(value_name='missing'), y='variable', hue='missing', multiple='fill', aspect=1.25)
plt.figure(figsize=(10, 6))
sns.heatmap(_input1.isna().transpose(), cmap='YlGnBu', cbar_kws={'label': 'Missing Data'})
X = _input1.drop(['Transported', 'PassengerId', 'Name', 'Cabin', 'Destination'], axis=1)
y = _input1['Transported']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = ['HomePlanet', 'CryoSleep', 'VIP']
for i in num_cols:
    X[i] = X[i].fillna(np.mean(X[i]), inplace=False)
for j in cat_cols:
    X[j] = X[j].fillna(X[i].mode()[0], inplace=False)
plt.figure(figsize=(10, 6))
sns.displot(data=X.isna().melt(value_name='missing'), y='variable', hue='missing', multiple='fill', aspect=1.25)
le = preprocessing.LabelEncoder()
for i in num_cols:
    X[i] = pd.qcut(X[i], q=5, duplicates='drop')
    X[i] = le.fit_transform(X[i])
X = pd.get_dummies(X, columns=['HomePlanet', 'CryoSleep', 'VIP'], drop_first=True)
X.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.svm import SVC
svc = SVC(kernel='rbf')