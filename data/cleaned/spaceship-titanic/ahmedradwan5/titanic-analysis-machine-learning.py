import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, RocCurveDisplay, auc, roc_curve
plt.rcParams['figure.figsize'] = (9, 7)
plt.rcParams['font.size'] = 14
sns.set_style('whitegrid')
sns.set_theme(style='ticks')
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train.head()
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_test
df_train.info()
df_train = df_train.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
df_train.head()
for i in df_train.columns:
    print(f'Unique values in the {i} column = ', df_train[i].unique())
    print('----------------------------------------------------------------------------')
df_train[df_train['HomePlanet'].isna()]
df_train = df_train.dropna()
df_train.info()
df_train.corr()
sns.heatmap(df_train.corr(), annot=True)
for i in df_test.columns:
    print(f'Unique values in the {i} column = ', df_test[i].unique())
    print('----------------------------------------------------------------------------')
df_test = df_test.drop(['PassengerId', 'Name'], axis=1)
df_test.info()
from pandas_profiling import ProfileReport
profile = ProfileReport(df_train, title='Pandas Profiling Report')
profile.to_file('your_report_project.html')
df_train.head()
from sklearn.preprocessing import LabelEncoder
lE = LabelEncoder()
df_train['HomePlanet'] = lE.fit_transform(df_train['HomePlanet'])
df_train['Destination'] = lE.fit_transform(df_train['Destination'])
df_train['CryoSleep'] = pd.get_dummies(df_train['CryoSleep'], drop_first=True)
df_train['VIP'] = pd.get_dummies(df_train['VIP'], drop_first=True)
df_train['Transported'] = pd.get_dummies(df_train['Transported'], drop_first=True)
df_train.head()
df_train.info()
for i in df_train.columns:
    print(f'Unique values in the {i} column = ', df_train[i].unique())
    print('----------------------------------------------------------------------------')
df_test.head()
df_test = df_test.drop(['Cabin'], axis=1)
df_test['HomePlanet'] = lE.fit_transform(df_test['HomePlanet'])
df_test['Destination'] = lE.fit_transform(df_test['Destination'])
df_test['Destination'].value_counts()
df_test['CryoSleep'] = pd.get_dummies(df_test['CryoSleep'], drop_first=True)
df_test['VIP'] = pd.get_dummies(df_test['VIP'], drop_first=True)
df_test.head()
df_test.info()
df_test = df_test.dropna()
X = df_train.drop('Transported', axis=1)
y = df_train['Transported']
X
y
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
LRC = LogisticRegression()