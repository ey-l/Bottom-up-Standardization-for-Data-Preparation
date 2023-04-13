import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.columns
df.columns
df.dtypes
df.info()
df.describe()
df = df.drop_duplicates()
df.isnull().sum()
print(df[df['BloodPressure'] == 0].shape[0])
print(df[df['Glucose'] == 0].shape[0])
print(df[df['SkinThickness'] == 0].shape[0])
print(df[df['Insulin'] == 0].shape[0])
print(df[df['BMI'] == 0].shape[0])
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())
df.hist(bins=10, figsize=(10, 10))
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
from pandas.plotting import scatter_matrix
scatter_matrix(df, figsize=(20, 20))
corrmat = df.corr()
pass
df_selected = df.drop(['BloodPressure', 'Insulin', 'DiabetesPedigreeFunction'], axis='columns')
from sklearn.preprocessing import QuantileTransformer
x = df_selected
quantile = QuantileTransformer()
X = quantile.fit_transform(x)
df_new = quantile.transform(X)
df_new = pd.DataFrame(X)
df_new.columns = ['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age', 'Outcome']
df_new.head()
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
target_name = 'Outcome'
y = df_new[target_name]
X = df_new.drop(target_name, axis=1)
X.head()
y.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
(X_train.shape, y_train.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()
n_neighbors = list(range(15, 25))
p = [1, 2]
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
hyperparameters = dict(n_neighbors=n_neighbors, p=p, weights=weights, metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=knn, param_grid=hyperparameters, n_jobs=-1, cv=cv, scoring='f1', error_score=0)