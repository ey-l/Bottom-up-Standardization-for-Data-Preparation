import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import seaborn as sns
import pickle
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
ProfileReport(df)
count = (df['Glucose'] == 0).sum()
print('Count of zeros in Column  Glucose : ', count)
count = (df['BMI'] == 0).sum()
print('Count of zeros in Column  BMI : ', count)
count = (df['SkinThickness'] == 0).sum()
print('Count of zeros in Column  SkinThickness : ', count)
count = (df['Age'] == 0).sum()
print('Count of zeros in Column  Age : ', count)
count = (df['BloodPressure'] == 0).sum()
print('Count of zeros in Column  BloodPressure : ', count)
count = (df['Insulin'] == 0).sum()
print('Count of zeros in Column  Insulin : ', count)
count = (df['DiabetesPedigreeFunction'] == 0).sum()
print('Count of zeros in Column  DiabetesPedigreeFunction : ', count)
count = (df['Pregnancies'] == 0).sum()
print('Count of zeros in Column  Pregnancies : ', count)
df['BMI'] = df['BMI'].replace(0, df['BMI'].mean())
df.columns
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].mean())
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].mean())
ProfileReport(df)
pass
pass
q = df['Insulin'].quantile(0.95)
df_new = df[df['Insulin'] < q]
df_new
pass
pass
q = df['Pregnancies'].quantile(0.98)
df_new = df_new[df_new['Pregnancies'] < q]
q = df_new['BMI'].quantile(0.99)
df_new = df_new[df_new['BMI'] < q]
q = df_new['SkinThickness'].quantile(0.99)
df_new = df_new[df_new['SkinThickness'] < q]
q = df_new['Insulin'].quantile(0.95)
df_new = df_new[df_new['Insulin'] < q]
q = df_new['DiabetesPedigreeFunction'].quantile(0.99)
df_new = df_new[df_new['DiabetesPedigreeFunction'] < q]
q = df_new['Age'].quantile(0.99)
df_new = df_new[df_new['Age'] < q]
pass
pass
ProfileReport(df_new)
y = df_new['Outcome']
y
X = df_new.drop(columns=['Outcome'])
X
scalar = StandardScaler()
ProfileReport(pd.DataFrame(scalar.fit_transform(X)))
X_scaled = scalar.fit_transform(X)
df_new_scalar = pd.DataFrame(scalar.fit_transform(df_new))
pass
pass
X_scaled
y

def vif_score(X):
    scalar = StandardScaler()
    arr = scalar.fit_transform(X)
    return pd.DataFrame([[X.columns[i], variance_inflation_factor(arr, i)] for i in range(arr.shape[1])], columns=['FEATURE', 'VIF_SCORE'])
vif_score(X)
(X_train, X_test, y_train, y_test) = train_test_split(X_scaled, y, test_size=0.2, random_state=144)
X_train
X_test
logr_liblinear = LogisticRegression(verbose=1, solver='liblinear')