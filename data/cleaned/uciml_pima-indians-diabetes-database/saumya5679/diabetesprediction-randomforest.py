import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.shape
data.columns.tolist()
data.dtypes
data.isnull().sum()
data.info()
data.describe()
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
data.isnull().sum()
data['Glucose'].fillna(data['Glucose'].median(), inplace=True)
data['BloodPressure'].fillna(data['BloodPressure'].median(), inplace=True)
data['SkinThickness'].fillna(data['SkinThickness'].median(), inplace=True)
data['Insulin'].fillna(data['Insulin'].median(), inplace=True)
data['BMI'].fillna(data['BMI'].mean(), inplace=True)
data.isnull().sum()
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15, 20))
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap='RdYlGn')
data.corr()
data['Pregnancies'].unique()
data['Pregnancies'].value_counts().sort_values()
data.groupby('Outcome')[['Pregnancies', 'Glucose', 'BloodPressure']].agg(['max', 'min', 'mean'])
data.groupby('Outcome')[['SkinThickness', 'Insulin', 'BMI', 'Age']].agg(['max', 'min', 'mean'])
data['Outcome'].value_counts()
p = data.hist(figsize=(15, 20))
X = data.drop('Outcome', axis=1)
X.head()
y = data['Outcome']
y.head()
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=200)
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
clf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=100)
clf
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')
scores.mean()