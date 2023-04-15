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
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train.head()
df_train.info()
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
id_pass = df_test['PassengerId']
df_train.describe()
df_train.shape
df_train.isnull().sum()
print('column \t       Missing Percentage  ')
print('=' * 30)
for col in df_train.columns:
    percentage_col = df_train[col].isnull().sum() / len(df_train) * 100
    print('{}\t{} % '.format(col.ljust(15, ' '), round(percentage_col, 3)))
df_train['CryoSleep'].mode()
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
df_train['RoomService'] = df_train['RoomService'].fillna(df_train['RoomService'].mean())
df_train['FoodCourt'] = df_train['FoodCourt'].fillna(df_train['FoodCourt'].mean())
df_train['ShoppingMall'] = df_train['ShoppingMall'].fillna(df_train['ShoppingMall'].mean())
df_train['Spa'] = df_train['Spa'].fillna(df_train['Spa'].mean())
df_train['VRDeck'] = df_train['VRDeck'].fillna(df_train['VRDeck'].mean())
df_train['HomePlanet'] = df_train['HomePlanet'].fillna('Earth')
df_train['CryoSleep'] = df_train['CryoSleep'].fillna('False')
df_train['Cabin'] = df_train['Cabin'].fillna('G/734/S')
df_train['Destination'] = df_train['Destination'].fillna('TRAPPIST-1e')
df_train['VIP'] = df_train['VIP'].fillna('False')
df_train = df_train.drop(['PassengerId', 'Name'], axis=1)
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())
df_test['RoomService'] = df_test['RoomService'].fillna(df_test['RoomService'].mean())
df_test['FoodCourt'] = df_test['FoodCourt'].fillna(df_test['FoodCourt'].mean())
df_test['ShoppingMall'] = df_test['ShoppingMall'].fillna(df_test['ShoppingMall'].mean())
df_test['Spa'] = df_test['Spa'].fillna(df_test['Spa'].mean())
df_test['VRDeck'] = df_test['VRDeck'].fillna(df_test['VRDeck'].mean())
df_test['HomePlanet'] = df_test['HomePlanet'].fillna('Earth')
df_test['CryoSleep'] = df_test['CryoSleep'].fillna('False')
df_test['Cabin'] = df_test['Cabin'].fillna('G/734/S')
df_test['Destination'] = df_test['Destination'].fillna('TRAPPIST-1e')
df_test['VIP'] = df_test['VIP'].fillna('False')
df_test = df_test.drop(['PassengerId', 'Name'], axis=1)
df_train.isnull().sum()
plt.rcParams['figure.figsize'] = (18, 12)
plt.subplot(2, 4, 1)
sns.distplot(df_train['Age'], color='red')
plt.grid()
plt.subplot(2, 4, 2)
sns.distplot(df_train['RoomService'], color='black')
plt.grid()
plt.subplot(2, 4, 3)
sns.distplot(df_train['FoodCourt'], color='black')
plt.grid()
plt.subplot(2, 4, 4)
sns.distplot(df_train['Spa'], color='red')
plt.grid()
plt.subplot(2, 4, 5)
sns.distplot(df_train['ShoppingMall'], color='red')
plt.grid()
plt.subplot(2, 4, 6)
sns.distplot(df_train['VRDeck'], color='red')
plt.grid()
plt.rcParams['figure.figsize'] = (18, 12)
plt.subplot(2, 4, 1)
sns.distplot(df_test['Age'], color='red')
plt.grid()
plt.subplot(2, 4, 2)
sns.distplot(df_test['RoomService'], color='black')
plt.grid()
plt.subplot(2, 4, 3)
sns.distplot(df_test['FoodCourt'], color='black')
plt.grid()
plt.subplot(2, 4, 4)
sns.distplot(df_test['Spa'], color='red')
plt.grid()
plt.subplot(2, 4, 5)
sns.distplot(df_test['ShoppingMall'], color='red')
plt.grid()
plt.subplot(2, 4, 6)
sns.distplot(df_test['VRDeck'], color='red')
plt.grid()
sns.catplot(y='HomePlanet', hue='Transported', kind='count', palette='pastel', edgecolor='.6', data=df_train)
sns.catplot(y='CryoSleep', hue='Transported', kind='count', palette='pastel', edgecolor='.6', data=df_train)
sns.catplot(y='Destination', hue='Transported', kind='count', palette='pastel', edgecolor='.6', data=df_train)
sns.catplot(y='VIP', hue='Transported', kind='count', palette='pastel', edgecolor='.6', data=df_train)
df_train.isnull().sum()
df_train['Age1'] = age_band(df_train)
df_test['Age1'] = age_band(df_test)
for i in df_train.columns:
    if df_train[i].dtype == 'O':
        continue
    df_train[i] = np.sqrt(np.sqrt(df_train[i]))
for i in df_test.columns:
    if df_test[i].dtype == 'O':
        continue
    df_test[i] = np.sqrt(np.sqrt(df_test[i]))
plt.rcParams['figure.figsize'] = (18, 12)
plt.subplot(2, 4, 1)
sns.distplot(df_test['Age'], color='red')
plt.grid()
plt.subplot(2, 4, 2)
sns.distplot(df_test['RoomService'], color='black')
plt.grid()
plt.subplot(2, 4, 3)
sns.distplot(df_test['FoodCourt'], color='black')
plt.grid()
plt.subplot(2, 4, 4)
sns.distplot(df_test['Spa'], color='red')
plt.grid()
plt.subplot(2, 4, 5)
sns.distplot(df_test['ShoppingMall'], color='red')
plt.grid()
plt.subplot(2, 4, 6)
sns.distplot(df_test['VRDeck'], color='red')
plt.grid()
df_train['Destination'].unique()
df_train['Age1'].value_counts()
df_train = df_train.astype({'CryoSleep': 'bool', 'VIP': 'bool'})
df_test = df_test.astype({'CryoSleep': 'bool', 'VIP': 'bool'})
bool_map = {True: 1, False: 0}
planet = {'Europa': 1, 'Earth': 2, 'Mars': 3}
dest = {'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}
age = {'Adult': 1, 'Adolescence': 2, 'Child': 3, 'Senior': 4}
df_train.head(1)
df_train['HomePlanet'] = df_train['HomePlanet'].map(planet)
df_train['CryoSleep'] = df_train['CryoSleep'].map(bool_map)
df_train['Destination'] = df_train['Destination'].map(dest)
df_train['VIP'] = df_train['VIP'].map(bool_map)
df_train['Age1'] = df_train['Age1'].map(age)
df_test['HomePlanet'] = df_test['HomePlanet'].map(planet)
df_test['CryoSleep'] = df_test['CryoSleep'].map(bool_map)
df_test['Destination'] = df_test['Destination'].map(dest)
df_test['VIP'] = df_test['VIP'].map(bool_map)
df_test['Age1'] = df_test['Age1'].map(age)
corr = df_train.corr()
sns.heatmap(corr, annot=True)
corr['Transported']
df_train = df_train.drop('Cabin', axis=1)
df_test = df_test.drop('Cabin', axis=1)
from sklearn.model_selection import train_test_split
X = df_train.drop('Transported', axis=1)
y = df_train['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25)
df_train.isnull().sum()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier()
forest_params = [{'max_depth': list(range(10, 15)), 'max_features': list(range(0, 22))}]
clf = GridSearchCV(rfc, forest_params, cv=10, scoring='accuracy')