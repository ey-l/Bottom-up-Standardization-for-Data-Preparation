import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
pima_column_names = ['times_pregnant', 'plasma_glucose_concentration', 'diastolic_blood_pressure', 'triceps_thickness', 'serum_insulin', 'bmi', 'pedigree_function', 'age', 'onset_diabetes']
pima = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv', names=pima_column_names, skiprows=1)
pima.head()
pima.info()
pima['onset_diabetes'].value_counts(normalize=True)
col = 'plasma_glucose_concentration'
plt.figure(figsize=(10, 5))
plt.hist(pima[pima['onset_diabetes'] == 0][col], 10, alpha=0.5, label='non-diabetes')
plt.hist(pima[pima['onset_diabetes'] == 1][col], 10, alpha=0.5, label='diabetes')
plt.legend(loc='upper right')
plt.xlabel(col)
plt.ylabel('Frequency')
plt.title('Histogram of {}'.format(col))

for col in ['bmi', 'diastolic_blood_pressure', 'serum_insulin', 'triceps_thickness', 'plasma_glucose_concentration']:
    plt.figure(figsize=(8, 4))
    plt.hist(pima[pima['onset_diabetes'] == 0][col], 10, alpha=0.5, label='non-diabetes')
    plt.hist(pima[pima['onset_diabetes'] == 1][col], 10, alpha=0.5, label='diabetes')
    plt.legend(loc='upper right')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title('Histogram of {}'.format(col))

plt.figure(figsize=(12, 8))
corr = round(pima.corr(), 2)
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.heatmap(corr, mask=mask, square=True, annot=True)
plt.xticks(rotation=90)

pima.corr()['onset_diabetes']
pima.describe()
pima['serum_insulin'] = pima['serum_insulin'].map(lambda x: x if x != 0 else None)
pima['serum_insulin'].isnull().sum()
pima.describe()
columns = ['bmi', 'plasma_glucose_concentration', 'diastolic_blood_pressure', 'triceps_thickness']
for col in columns:
    pima[col] = pima[col].map(lambda x: x if x != 0 else None)
pima.isnull().sum()
pima.info()
pima.describe()
pima.head(5)
(pima['plasma_glucose_concentration'].mean(), pima['plasma_glucose_concentration'].std())
empty_plasma_index = pima[pima['plasma_glucose_concentration'].isnull()].index
pima.loc[empty_plasma_index]['plasma_glucose_concentration']

def relation_with_output(column):
    temp = pima[pima[column].notnull()]
    d = temp[[column, 'onset_diabetes']].groupby(['onset_diabetes'])[column].apply(lambda x: x.median()).reset_index()
    return d
relation_with_output('plasma_glucose_concentration')
relation_with_output('diastolic_blood_pressure')
relation_with_output('triceps_thickness')
relation_with_output('serum_insulin')
relation_with_output('bmi')
pima.isnull().sum()
pima.loc[(pima['onset_diabetes'] == 0) & pima['serum_insulin'].isnull(), 'serum_insulin'] = 102.5
pima.loc[(pima['onset_diabetes'] == 1) & pima['serum_insulin'].isnull(), 'serum_insulin'] = 169.5
pima.loc[(pima['onset_diabetes'] == 0) & pima['bmi'].isnull(), 'bmi'] = 30.1
pima.loc[(pima['onset_diabetes'] == 1) & pima['bmi'].isnull(), 'bmi'] = 34.3
pima.loc[(pima['onset_diabetes'] == 0) & pima['triceps_thickness'].isnull(), 'triceps_thickness'] = 27.0
pima.loc[(pima['onset_diabetes'] == 1) & pima['triceps_thickness'].isnull(), 'triceps_thickness'] = 32.0
pima.loc[(pima['onset_diabetes'] == 0) & pima['diastolic_blood_pressure'].isnull(), 'diastolic_blood_pressure'] = 70.0
pima.loc[(pima['onset_diabetes'] == 1) & pima['diastolic_blood_pressure'].isnull(), 'diastolic_blood_pressure'] = 75.0
pima.loc[(pima['onset_diabetes'] == 0) & pima['plasma_glucose_concentration'].isnull(), 'plasma_glucose_concentration'] = 107.0
pima.loc[(pima['onset_diabetes'] == 1) & pima['plasma_glucose_concentration'].isnull(), 'plasma_glucose_concentration'] = 140.0
pima.isnull().sum()
X = pima.loc[:, :'age']
y = pima['onset_diabetes']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=99)
knn = KNeighborsClassifier()