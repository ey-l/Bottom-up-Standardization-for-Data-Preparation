import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pandas_profiling import ProfileReport
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train.head()
ProfileReport(train, title='EDA Report Spaceship Titanic')
round(train.isna().sum() / len(train) * 100, 2)
plt.figure(figsize=(10, 6))
sns.displot(data=train.isna().melt(value_name='missing'), y='variable', hue='missing', multiple='fill', aspect=1.25)
plt.figure(figsize=(10, 6))
sns.heatmap(train.isna().transpose(), cmap='YlGnBu', cbar_kws={'label': 'Missing Data'})
X = train.drop(['Transported', 'PassengerId', 'Name', 'Cabin', 'Destination'], axis=1)
y = train['Transported']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = ['HomePlanet', 'CryoSleep', 'VIP']
for i in num_cols:
    X[i].fillna(np.mean(X[i]), inplace=True)
for j in cat_cols:
    X[j].fillna(X[i].mode()[0], inplace=True)
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