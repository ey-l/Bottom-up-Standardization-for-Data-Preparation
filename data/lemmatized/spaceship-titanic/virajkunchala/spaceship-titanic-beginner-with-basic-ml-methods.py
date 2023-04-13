import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
_input1.describe()
_input0.describe()
_input1.dtypes
_input0.dtypes
categorical_features = []
numerical_features = []
for col in _input1.columns:
    if _input1[col].dtype == 'object':
        categorical_features.append(col)
    elif _input1[col].dtype == 'int' or _input1[col].dtype == 'float':
        numerical_features.append(col)
print(numerical_features)
print(categorical_features)
for x in numerical_features:
    _input0[x] = _input0[x].fillna(_input0[x].median())
    _input1[x] = _input1[x].fillna(_input1[x].median())
_input0.groupby(['HomePlanet', 'Cabin']).agg({'Cabin': 'count'})
_input0['HomePlanet'].isna().sum()
_input1.groupby(['HomePlanet', 'Cabin']).agg({'Cabin': 'count'})
tg = _input1['Cabin'].str[4:5]
tg
rt = _input1['Cabin'].str[:1]
_input1['Cabin'] = _input1['Cabin'].fillna('$', inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna('$', inplace=False)
_input1['nr_train'] = _input1['Cabin'].apply(lambda x: x[:1])
_input0['nr_test'] = _input0['Cabin'].apply(lambda x: x[:1])
_input1['mr_train'] = _input1['Cabin'].apply(lambda x: x[-1:])
_input0['mr_test'] = _input0['Cabin'].apply(lambda x: x[-1:])
_input1.head(3)
_input0.head(3)
_input1['nr_train'] = _input1['nr_train'].astype('object')
_input1.info()
_input1.groupby(['HomePlanet', 'nr_train']).agg({'nr_train': 'count'})
_input0.groupby(['HomePlanet', 'nr_test']).agg({'nr_test': 'count'})
if _input1['HomePlanet'] is np.nan and _input1['nr_train'] == 'G':
    _input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth')
elif _input1['HomePlanet'] is np.nan and _input1['nr_train'] in ('A', 'B', 'C', 'T'):
    _input1['HomePlanet'] = _input1['HomePlanet'].fillna('Europa')
if _input0['HomePlanet'] is np.nan and _input0['nr_test'] == 'G':
    _input0['HomePlanet'] = _input0['HomePlanet'].fillna('Earth')
elif _input0['HomePlanet'] is np.nan and _input0['nr_test'] in ('A', 'B', 'C', 'T'):
    _input0['HomePlanet'] = _input0['HomePlanet'].fillna('Europa')
_input1['HomePlanet'].isna().sum()
_input1['nr_train'] = _input1['nr_train'].replace('$', _input1['nr_train'].mode()[0])
_input0['nr_test'] = _input0['nr_test'].replace('$', _input0['nr_test'].mode()[0])
_input1['mr_train'] = _input1['mr_train'].replace('$', _input1['mr_train'].mode()[0])
_input0['mr_test'] = _input0['mr_test'].replace('$', _input0['mr_test'].mode()[0])
_input1[_input1['mr_train'] == '$'].count()
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(_input1['CryoSleep'].mode()[0])
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(_input0['CryoSleep'].mode()[0])
_input1['Destination'] = _input1['Destination'].fillna(_input1['Destination'].mode()[0])
_input0['Destination'] = _input0['Destination'].fillna(_input0['Destination'].mode()[0])
_input1['VIP'] = _input1['VIP'].fillna(_input1['VIP'].mode()[0])
_input0['VIP'] = _input0['VIP'].fillna(_input0['VIP'].mode()[0])
_input1['VIP'] = _input1['VIP'].astype(int)
_input0['VIP'] = _input0['VIP'].astype(int)
_input1['CryoSleep'] = _input1['CryoSleep'].map({True: 1, False: 0})
_input0['CryoSleep'] = _input0['CryoSleep'].map({True: 1, False: 0})
_input0 = _input0.drop(columns={'Name'}, inplace=False)
_input1 = _input1.drop(columns={'Name'}, inplace=False)
_input0 = _input0.drop(columns={'Cabin'}, inplace=False)
_input1 = _input1.drop(columns={'Cabin'}, inplace=False)
_input1.isna().sum()
_input0.isna().sum()
for col in numerical_features:
    sns.boxplot(_input1[col])
    sns.distplot(_input1[col])
for col in numerical_features:
    sns.distplot(_input1[col])
(fig, axs) = plt.subplots(1, 1, figsize=(10, 5))
sns.heatmap(_input1.corr(), cmap='Blues', annot=True)
from statsmodels.stats.outliers_influence import variance_inflation_factor
col_list = []
for col in _input1.columns:
    if (_input1[col].dtype != 'object') & (col != 'Transported'):
        col_list.append(col)
X = _input1[col_list]
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
vif_data
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
models = {'Logistic': LogisticRegression(), 'Decision Tree': DecisionTreeClassifier(), 'Random Forest': RandomForestClassifier()}
_input0.head(3)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
_input1['HomePlanet'] = labelencoder.fit_transform(_input1['HomePlanet'])
_input0['HomePlanet'] = labelencoder.fit_transform(_input0['HomePlanet'])
_input1['Destination'] = labelencoder.fit_transform(_input1['Destination'])
_input0['Destination'] = labelencoder.fit_transform(_input0['Destination'])
_input1['nr_train'] = labelencoder.fit_transform(_input1['nr_train'])
_input0['nr_test'] = labelencoder.fit_transform(_input0['nr_test'])
_input1['mr_train'] = labelencoder.fit_transform(_input1['mr_train'])
_input0['mr_test'] = labelencoder.fit_transform(_input0['mr_test'])
_input1.info()
_input1['Transported'] = _input1['Transported'].replace({True: 1, False: 0})
_input1[_input1['mr_train'] == '$'].count()
x = _input1.drop('Transported', axis=1)
x
y = _input1.iloc[:, -3]
y
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=45)
for x in range(len(list(models))):
    model = list(models.values())[x]