import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def age_band(df):
    l = []
    for i in range(len(df)):
        if df['Age'][i] >= 0 and df['Age'][i] <= 12:
            l.append('Child')
        if df['Age'][i] > 12 and df['Age'][i] <= 19:
            l.append('Adolescence')
        if df['Age'][i] > 19 and df['Age'][i] <= 59:
            l.append('Adult')
        if df['Age'][i] > 59:
            l.append('Senior')
    return l

def skew(df):
    print('Column \t       Skewness  ')
    print('=' * 30)
    for i in df.columns:
        if df[i].dtype == 'O':
            continue
        print('{}\t{}'.format(i.ljust(15, ' '), round(df[i].skew(), 3)))

def cal_accuracy(y_test, y_pred):
    print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))
    print('Accuracy : ', accuracy_score(y_test, y_pred) * 100)
    print('Report : ', classification_report(y_test, y_pred))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.info()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
id_pass = _input0['PassengerId']
_input1.describe()
_input1.shape
_input1.isnull().sum()
print('column \t       Missing Percentage  ')
print('=' * 30)
for col in _input1.columns:
    percentage_col = _input1[col].isnull().sum() / len(_input1) * 100
    print('{}\t{} % '.format(col.ljust(15, ' '), round(percentage_col, 3)))
_input1['CryoSleep'].mode()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean())
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].mean())
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].mean())
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].mean())
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].mean())
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].mean())
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth')
_input1['CryoSleep'] = _input1['CryoSleep'].fillna('False')
_input1['Cabin'] = _input1['Cabin'].fillna('G/734/S')
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e')
_input1['VIP'] = _input1['VIP'].fillna('False')
_input1 = _input1.drop(['PassengerId', 'Name'], axis=1)
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean())
_input0['RoomService'] = _input0['RoomService'].fillna(_input0['RoomService'].mean())
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(_input0['FoodCourt'].mean())
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(_input0['ShoppingMall'].mean())
_input0['Spa'] = _input0['Spa'].fillna(_input0['Spa'].mean())
_input0['VRDeck'] = _input0['VRDeck'].fillna(_input0['VRDeck'].mean())
_input0['HomePlanet'] = _input0['HomePlanet'].fillna('Earth')
_input0['CryoSleep'] = _input0['CryoSleep'].fillna('False')
_input0['Cabin'] = _input0['Cabin'].fillna('G/734/S')
_input0['Destination'] = _input0['Destination'].fillna('TRAPPIST-1e')
_input0['VIP'] = _input0['VIP'].fillna('False')
_input0 = _input0.drop(['PassengerId', 'Name'], axis=1)
_input1.isnull().sum()
plt.rcParams['figure.figsize'] = (18, 12)
plt.subplot(2, 4, 1)
sns.distplot(_input1['Age'], color='red')
plt.grid()
plt.subplot(2, 4, 2)
sns.distplot(_input1['RoomService'], color='black')
plt.grid()
plt.subplot(2, 4, 3)
sns.distplot(_input1['FoodCourt'], color='black')
plt.grid()
plt.subplot(2, 4, 4)
sns.distplot(_input1['Spa'], color='red')
plt.grid()
plt.subplot(2, 4, 5)
sns.distplot(_input1['ShoppingMall'], color='red')
plt.grid()
plt.subplot(2, 4, 6)
sns.distplot(_input1['VRDeck'], color='red')
plt.grid()
plt.rcParams['figure.figsize'] = (18, 12)
plt.subplot(2, 4, 1)
sns.distplot(_input0['Age'], color='red')
plt.grid()
plt.subplot(2, 4, 2)
sns.distplot(_input0['RoomService'], color='black')
plt.grid()
plt.subplot(2, 4, 3)
sns.distplot(_input0['FoodCourt'], color='black')
plt.grid()
plt.subplot(2, 4, 4)
sns.distplot(_input0['Spa'], color='red')
plt.grid()
plt.subplot(2, 4, 5)
sns.distplot(_input0['ShoppingMall'], color='red')
plt.grid()
plt.subplot(2, 4, 6)
sns.distplot(_input0['VRDeck'], color='red')
plt.grid()
sns.catplot(y='HomePlanet', hue='Transported', kind='count', palette='pastel', edgecolor='.6', data=_input1)
sns.catplot(y='CryoSleep', hue='Transported', kind='count', palette='pastel', edgecolor='.6', data=_input1)
sns.catplot(y='Destination', hue='Transported', kind='count', palette='pastel', edgecolor='.6', data=_input1)
sns.catplot(y='VIP', hue='Transported', kind='count', palette='pastel', edgecolor='.6', data=_input1)
_input1.isnull().sum()
_input1['Age1'] = age_band(_input1)
_input0['Age1'] = age_band(_input0)
for i in _input1.columns:
    if _input1[i].dtype == 'O':
        continue
    _input1[i] = np.sqrt(np.sqrt(_input1[i]))
for i in _input0.columns:
    if _input0[i].dtype == 'O':
        continue
    _input0[i] = np.sqrt(np.sqrt(_input0[i]))
plt.rcParams['figure.figsize'] = (18, 12)
plt.subplot(2, 4, 1)
sns.distplot(_input0['Age'], color='red')
plt.grid()
plt.subplot(2, 4, 2)
sns.distplot(_input0['RoomService'], color='black')
plt.grid()
plt.subplot(2, 4, 3)
sns.distplot(_input0['FoodCourt'], color='black')
plt.grid()
plt.subplot(2, 4, 4)
sns.distplot(_input0['Spa'], color='red')
plt.grid()
plt.subplot(2, 4, 5)
sns.distplot(_input0['ShoppingMall'], color='red')
plt.grid()
plt.subplot(2, 4, 6)
sns.distplot(_input0['VRDeck'], color='red')
plt.grid()
_input1['Destination'].unique()
_input1['Age1'].value_counts()
_input1 = _input1.astype({'CryoSleep': 'bool', 'VIP': 'bool'})
_input0 = _input0.astype({'CryoSleep': 'bool', 'VIP': 'bool'})
bool_map = {True: 1, False: 0}
planet = {'Europa': 1, 'Earth': 2, 'Mars': 3}
dest = {'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}
age = {'Adult': 1, 'Adolescence': 2, 'Child': 3, 'Senior': 4}
_input1.head(1)
_input1['HomePlanet'] = _input1['HomePlanet'].map(planet)
_input1['CryoSleep'] = _input1['CryoSleep'].map(bool_map)
_input1['Destination'] = _input1['Destination'].map(dest)
_input1['VIP'] = _input1['VIP'].map(bool_map)
_input1['Age1'] = _input1['Age1'].map(age)
_input0['HomePlanet'] = _input0['HomePlanet'].map(planet)
_input0['CryoSleep'] = _input0['CryoSleep'].map(bool_map)
_input0['Destination'] = _input0['Destination'].map(dest)
_input0['VIP'] = _input0['VIP'].map(bool_map)
_input0['Age1'] = _input0['Age1'].map(age)
corr = _input1.corr()
sns.heatmap(corr, annot=True)
corr['Transported']
_input1 = _input1.drop('Cabin', axis=1)
_input0 = _input0.drop('Cabin', axis=1)
from sklearn.model_selection import train_test_split
X = _input1.drop('Transported', axis=1)
y = _input1['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25)
_input1.isnull().sum()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier()
forest_params = [{'max_depth': list(range(10, 15)), 'max_features': list(range(0, 22))}]
clf = GridSearchCV(rfc, forest_params, cv=10, scoring='accuracy')