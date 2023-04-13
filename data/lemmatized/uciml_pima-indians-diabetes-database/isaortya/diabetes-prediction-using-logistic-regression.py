import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
BASE_DIR = Path('data/input/uciml_pima-indians-diabetes-database')
diabetes = pd.read_csv(BASE_DIR / 'diabetes.csv')
diabetes.head(10)
diabetes.info
features_list = list(diabetes.drop(columns='Outcome').columns)
columns = list(diabetes.columns)
print(features_list)
diabetes.dtypes
diabetes.isnull().sum()
diabetes[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(diabetes.isnull().sum())
null_rows = diabetes[diabetes.isnull().any(axis=1)]
print(null_rows.head(10))
pregnancies = diabetes['Pregnancies'].dropna()
pass
age = diabetes['Age'].dropna()
pass
bmi = diabetes['BMI'].dropna()
pass

def make_bins(col_name, num_of_bins):
    bin_names = []
    for i in range(1, num_of_bins + 1):
        bin_names.append(i)
    pass
    cutoff_values[0] = cutoff_values[0] - 0.5
    temp_col = pd.cut(diabetes[col_name], cutoff_values, labels=bin_names)
    return pd.to_numeric(temp_col)
diabetes['BMI_bin'] = make_bins('BMI', 5)
diabetes['Age_bin'] = make_bins('Age', 7)
diabetes['Pregnancies_bin'] = make_bins('Pregnancies', 5)
print(diabetes.head(10))
diabetes.dtypes
from sklearn.model_selection import train_test_split
X = diabetes.drop(columns=['BMI', 'Pregnancies', 'Age', 'Outcome'])
y = diabetes.Outcome
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=1)
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
knn_imputer = KNNImputer(add_indicator=True)
scaler = StandardScaler()
logistic_model = LogisticRegression(random_state=1)
pipeline = Pipeline(steps=[('KNN_imputer', knn_imputer), ('scaler', scaler), ('model', logistic_model)])
from sklearn.model_selection import cross_val_score
knn_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print('Accuracy scores: \n')
print(knn_scores)