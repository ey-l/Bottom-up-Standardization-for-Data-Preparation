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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
_input1.info()
_input1 = _input1.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
_input1.head()
for i in _input1.columns:
    print(f'Unique values in the {i} column = ', _input1[i].unique())
    print('----------------------------------------------------------------------------')
_input1[_input1['HomePlanet'].isna()]
_input1 = _input1.dropna()
_input1.info()
_input1.corr()
sns.heatmap(_input1.corr(), annot=True)
for i in _input0.columns:
    print(f'Unique values in the {i} column = ', _input0[i].unique())
    print('----------------------------------------------------------------------------')
_input0 = _input0.drop(['PassengerId', 'Name'], axis=1)
_input0.info()
from pandas_profiling import ProfileReport
profile = ProfileReport(_input1, title='Pandas Profiling Report')
profile.to_file('your_report_project.html')
_input1.head()
from sklearn.preprocessing import LabelEncoder
lE = LabelEncoder()
_input1['HomePlanet'] = lE.fit_transform(_input1['HomePlanet'])
_input1['Destination'] = lE.fit_transform(_input1['Destination'])
_input1['CryoSleep'] = pd.get_dummies(_input1['CryoSleep'], drop_first=True)
_input1['VIP'] = pd.get_dummies(_input1['VIP'], drop_first=True)
_input1['Transported'] = pd.get_dummies(_input1['Transported'], drop_first=True)
_input1.head()
_input1.info()
for i in _input1.columns:
    print(f'Unique values in the {i} column = ', _input1[i].unique())
    print('----------------------------------------------------------------------------')
_input0.head()
_input0 = _input0.drop(['Cabin'], axis=1)
_input0['HomePlanet'] = lE.fit_transform(_input0['HomePlanet'])
_input0['Destination'] = lE.fit_transform(_input0['Destination'])
_input0['Destination'].value_counts()
_input0['CryoSleep'] = pd.get_dummies(_input0['CryoSleep'], drop_first=True)
_input0['VIP'] = pd.get_dummies(_input0['VIP'], drop_first=True)
_input0.head()
_input0.info()
_input0 = _input0.dropna()
X = _input1.drop('Transported', axis=1)
y = _input1['Transported']
X
y
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
LRC = LogisticRegression()