import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1
_input1.info()
_input1.columns
_input1.describe()
_input1.describe().T
_input0
sns.countplot(data=_input1, y='HomePlanet', hue='Destination')
plt.figure(figsize=(10, 7))
cor_matrix = _input1.loc[:, 'RoomService':'VRDeck']
sns.heatmap(cor_matrix.corr(), annot=True)
_input1['CabinSide'] = _input1.apply(lambda x: str(x['Cabin'])[-1:], axis=1)
_input1
imp_cols = _input1.columns[1:-2].tolist() + ['CabinSide']
imp_cols.remove('Cabin')
imp_cols.remove('Name')
tar_col = 'Transported'
print('Important columns :', imp_cols)
print('Target column :', tar_col)
X = _input1[imp_cols]
y = _input1[tar_col]
X
y
X.info()
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()
print('Numerical columns :', num_cols)
print('Categorical columns :', cat_cols)
X[num_cols]
X[cat_cols]
X[num_cols].isna().sum()
imputer = SimpleImputer()
X[num_cols] = imputer.fit_transform(X[num_cols])
X[num_cols].isna().sum()
X[num_cols].describe()
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X.describe()
X[cat_cols].nunique()
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')