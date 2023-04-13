import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.info()
df.describe().T
df.isnull().any()
for x in df.columns:
    print(x, len(df[df[x] == 0]))
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
print(df.isnull().sum())
p = df.hist(figsize=(15, 15))
df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
df['Insulin'].fillna(df['Insulin'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)
print(df.isnull().sum())
df.Outcome.value_counts()
pass
pass
pass
percent_have_diabetes = len(df[df.Outcome == 1]) / len(df) * 100
percent_havenot_diabetes = len(df[df.Outcome == 0]) / len(df) * 100
print('Percent of people that have diabetes is :', round(percent_have_diabetes, 2), '%')
print("Percent of people that haven't diabetes is :", round(percent_havenot_diabetes, 2), '%')
pass
pass
100 * df.corr()['Outcome'].sort_values()
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
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
from sklearn.preprocessing import PolynomialFeatures
feature = PolynomialFeatures(degree=3)
X = feature.fit_transform(X)
X.shape
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.15, random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

def models(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(solver='liblinear')