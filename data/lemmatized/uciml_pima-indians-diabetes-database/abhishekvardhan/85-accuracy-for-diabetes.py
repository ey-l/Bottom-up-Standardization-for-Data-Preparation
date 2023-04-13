import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df['Outcome'].value_counts().plot(kind='bar')
df1 = df.copy()
for i in df.columns:
    pass
    pass
    pass
df['ins'] = np.where(df['Insulin'] == 0, 1, 0)
print(df.groupby('ins')['Outcome'].mean())
df['sk'] = np.where(df['SkinThickness'] == 0, 1, 0)
print(df.groupby('sk')['Outcome'].mean())
df['Glucose'] = np.where(df['Glucose'] == 0, df['Glucose'].mean(), df['Glucose'])
df['BloodPressure'] = np.where(df['BloodPressure'] == 0, df['BloodPressure'].mean(), df['BloodPressure'])
df['BMI'] = np.where(df['BMI'] == 0, df['BMI'].mean(), df['BMI'])
for i in df.columns:
    pass
    pass
    pass
print(df1.groupby('Outcome')['SkinThickness'].mean())
df1 = df[df['SkinThickness'] != 0]
a = df1.groupby('Outcome')['SkinThickness'].mean()[0]
print(a)
b = df1.groupby('Outcome')['SkinThickness'].mean()[1]
df1 = df[df['Insulin'] != 0]
c = df1.groupby('Outcome')['Insulin'].mean()[0]
d = df1.groupby('Outcome')['Insulin'].mean()[1]
df.loc[(df['SkinThickness'] == 0) & (df['Outcome'] == 1), 'SkinThickness'] = b
df.loc[(df['SkinThickness'] == 0) & (df['Outcome'] == 0), 'SkinThickness'] = a
df.loc[(df['SkinThickness'] == 0) & (df['Outcome'] == 1), 'SkinThickness'] = d
df.loc[(df['SkinThickness'] == 0) & (df['Outcome'] == 0), 'SkinThickness'] = c
df
for i in df.columns:
    pass
    pass
    pass
pass
pass
pass
for fet in df.columns:
    pass
    pass
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
df = df[~((df < Q1 - 3 * IQR) | (df > Q3 + 3 * IQR)).any(axis=1)]
for fet in df.columns:
    pass
    pass
a = ['DiabetesPedigreeFunction', 'Age', 'Insulin', 'Pregnancies']

def diagnostic_plots(df, variable):
    pass
    pass
    df[variable].hist(bins=20)
    pass
    stats.probplot(df[variable], dist='norm', plot=plt)
    pass
for i in df.columns:
    diagnostic_plots(df, i)
for i in a:
    df[i] = np.log(df[i] + 1)
    diagnostic_plots(df, i)
x = df.drop('Outcome', axis=1)
y = df['Outcome']
from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler()
x = scl.fit_transform(df.drop('Outcome', axis=1))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
clf1 = LogisticRegression()
cross_val_score(clf1, x, y, cv=20).mean()
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=42)
from sklearn.model_selection import GridSearchCV
for i in [0.1, 0.5, 1, 2, 3, 5, 10, 50, 100]:
    clf2 = LogisticRegression(C=i)