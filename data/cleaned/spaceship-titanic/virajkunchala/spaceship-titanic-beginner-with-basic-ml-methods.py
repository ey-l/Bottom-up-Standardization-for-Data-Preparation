import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train.head()
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_test
df_train.describe()
df_test.describe()
df_train.dtypes
df_test.dtypes
categorical_features = []
numerical_features = []
for col in df_train.columns:
    if df_train[col].dtype == 'object':
        categorical_features.append(col)
    elif df_train[col].dtype == 'int' or df_train[col].dtype == 'float':
        numerical_features.append(col)
print(numerical_features)
print(categorical_features)
for x in numerical_features:
    df_test[x] = df_test[x].fillna(df_test[x].median())
    df_train[x] = df_train[x].fillna(df_train[x].median())
df_test.groupby(['HomePlanet', 'Cabin']).agg({'Cabin': 'count'})
df_test['HomePlanet'].isna().sum()
df_train.groupby(['HomePlanet', 'Cabin']).agg({'Cabin': 'count'})
tg = df_train['Cabin'].str[4:5]
tg
rt = df_train['Cabin'].str[:1]
df_train['Cabin'].fillna('$', inplace=True)
df_test['Cabin'].fillna('$', inplace=True)
df_train['nr_train'] = df_train['Cabin'].apply(lambda x: x[:1])
df_test['nr_test'] = df_test['Cabin'].apply(lambda x: x[:1])
df_train['mr_train'] = df_train['Cabin'].apply(lambda x: x[-1:])
df_test['mr_test'] = df_test['Cabin'].apply(lambda x: x[-1:])
df_train.head(3)
df_test.head(3)
df_train['nr_train'] = df_train['nr_train'].astype('object')
df_train.info()
df_train.groupby(['HomePlanet', 'nr_train']).agg({'nr_train': 'count'})
df_test.groupby(['HomePlanet', 'nr_test']).agg({'nr_test': 'count'})
if df_train['HomePlanet'] is np.nan and df_train['nr_train'] == 'G':
    df_train['HomePlanet'] = df_train['HomePlanet'].fillna('Earth')
elif df_train['HomePlanet'] is np.nan and df_train['nr_train'] in ('A', 'B', 'C', 'T'):
    df_train['HomePlanet'] = df_train['HomePlanet'].fillna('Europa')
if df_test['HomePlanet'] is np.nan and df_test['nr_test'] == 'G':
    df_test['HomePlanet'] = df_test['HomePlanet'].fillna('Earth')
elif df_test['HomePlanet'] is np.nan and df_test['nr_test'] in ('A', 'B', 'C', 'T'):
    df_test['HomePlanet'] = df_test['HomePlanet'].fillna('Europa')
df_train['HomePlanet'].isna().sum()
df_train['nr_train'] = df_train['nr_train'].replace('$', df_train['nr_train'].mode()[0])
df_test['nr_test'] = df_test['nr_test'].replace('$', df_test['nr_test'].mode()[0])
df_train['mr_train'] = df_train['mr_train'].replace('$', df_train['mr_train'].mode()[0])
df_test['mr_test'] = df_test['mr_test'].replace('$', df_test['mr_test'].mode()[0])
df_train[df_train['mr_train'] == '$'].count()
df_train['CryoSleep'] = df_train['CryoSleep'].fillna(df_train['CryoSleep'].mode()[0])
df_test['CryoSleep'] = df_test['CryoSleep'].fillna(df_test['CryoSleep'].mode()[0])
df_train['Destination'] = df_train['Destination'].fillna(df_train['Destination'].mode()[0])
df_test['Destination'] = df_test['Destination'].fillna(df_test['Destination'].mode()[0])
df_train['VIP'] = df_train['VIP'].fillna(df_train['VIP'].mode()[0])
df_test['VIP'] = df_test['VIP'].fillna(df_test['VIP'].mode()[0])
df_train['VIP'] = df_train['VIP'].astype(int)
df_test['VIP'] = df_test['VIP'].astype(int)
df_train['CryoSleep'] = df_train['CryoSleep'].map({True: 1, False: 0})
df_test['CryoSleep'] = df_test['CryoSleep'].map({True: 1, False: 0})
df_test.drop(columns={'Name'}, inplace=True)
df_train.drop(columns={'Name'}, inplace=True)
df_test.drop(columns={'Cabin'}, inplace=True)
df_train.drop(columns={'Cabin'}, inplace=True)
df_train.isna().sum()
df_test.isna().sum()
for col in numerical_features:
    sns.boxplot(df_train[col])

    sns.distplot(df_train[col])

for col in numerical_features:
    sns.distplot(df_train[col])

(fig, axs) = plt.subplots(1, 1, figsize=(10, 5))
sns.heatmap(df_train.corr(), cmap='Blues', annot=True)

from statsmodels.stats.outliers_influence import variance_inflation_factor
col_list = []
for col in df_train.columns:
    if (df_train[col].dtype != 'object') & (col != 'Transported'):
        col_list.append(col)
X = df_train[col_list]
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
df_test.head(3)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df_train['HomePlanet'] = labelencoder.fit_transform(df_train['HomePlanet'])
df_test['HomePlanet'] = labelencoder.fit_transform(df_test['HomePlanet'])
df_train['Destination'] = labelencoder.fit_transform(df_train['Destination'])
df_test['Destination'] = labelencoder.fit_transform(df_test['Destination'])
df_train['nr_train'] = labelencoder.fit_transform(df_train['nr_train'])
df_test['nr_test'] = labelencoder.fit_transform(df_test['nr_test'])
df_train['mr_train'] = labelencoder.fit_transform(df_train['mr_train'])
df_test['mr_test'] = labelencoder.fit_transform(df_test['mr_test'])
df_train.info()
df_train['Transported'] = df_train['Transported'].replace({True: 1, False: 0})
df_train[df_train['mr_train'] == '$'].count()
x = df_train.drop('Transported', axis=1)
x
y = df_train.iloc[:, -3]
y
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=45)
for x in range(len(list(models))):
    model = list(models.values())[x]