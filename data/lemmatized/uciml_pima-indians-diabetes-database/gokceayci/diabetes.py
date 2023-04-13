import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe().T
df.isnull().values.any()
df.eq(0).sum()
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.NaN)
df.fillna(df.mean(), inplace=True)
df.head()
df.isnull().sum()
df.eq(0).sum()
p = df.hist(bins=50, figsize=(20, 15))
df.plot(kind='density', subplots=True, layout=(3, 3), figsize=(20, 15), sharex=False)
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
pass
c = ['green', 'red']
print(df.Outcome.value_counts())
df.Outcome.value_counts().plot(kind='bar', color=c)
df.corr()
pass
pass
df.corr().nlargest(4, 'Outcome').index
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
x = df[['Glucose', 'BMI', 'Age']]
y = df.iloc[:, 8]
y
log_reg = linear_model.LogisticRegression()
log_reg_score = cross_val_score(log_reg, x, y, cv=10, scoring='accuracy').mean()
log_reg_score
from sklearn import svm
linear_svm = svm.SVC(kernel='linear')
linear_svem_score = cross_val_score(linear_svm, x, y, cv=10, scoring='accuracy').mean()
linear_svem_score
import pickle
filename = 'diabetes.sav'