import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from pandas_profiling import ProfileReport
import scipy.stats
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
from scipy.stats import shapiro
(stat, p) = shapiro(dataset['BloodPressure'])
print('Statistics=%.3f,p=%.3f' % (stat, p))
dataset.shape
dataset['Outcome'].value_counts()
pass
df1 = dataset.loc[dataset['Outcome'] == 1]
df2 = dataset.loc[dataset['Outcome'] == 0]
df1 = df1.replace({'BloodPressure': 0}, np.median(dataset['BloodPressure']))
df2 = df2.replace({'BloodPressue': 0}, np.median(dataset['BloodPressure']))
dataframe = [df1, df2]
dataset = pd.concat(dataframe)
dataset
df1.shape
pass
df1 = df1.replace({'BMI': 0}, np.median(df1['BMI']))
df2 = df2.replace({'BMI': 0}, np.median(df2['BMI']))
dataframe = [df1, df2]
dataset = pd.concat(dataframe)
dataset
df1 = df1.replace({'DiabetesPedigreeFunction': 0}, np.median(df1['DiabetesPedigreeFunction']))
df2 = df2.replace({'DiabetesPedigreeFunction': 0}, np.median(df1['DiabetesPedigreeFunction']))
dataframe = [df1, df2]
dataset = pd.concat(dataframe)
dataset
df1 = df1.replace({'Glucose': 0}, np.median(df1['Glucose']))
df2 = df2.replace({'Glucose': 0}, np.median(df2['Glucose']))
dataframe = [df1, df2]
dataset = pd.concat(dataframe)
dataset
df1 = df1.replace({'Insulin': 0}, np.mean(df1['Insulin']))
df2 = df2.replace({'Insulin': 0}, np.mean(df2['Insulin']))
dataframe = [df1, df2]
dataset = pd.concat(dataframe)
dataset
from scipy.stats import pearsonr
(corr, _) = pearsonr(dataset['Pregnancies'], dataset['Age'])
corr
X = dataset.drop('Outcome', axis=1).values
y = dataset['Outcome']
X
dataset.info()
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)
X
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3)
len(y_train)
pass
from imblearn.over_sampling import SMOTE
smt = SMOTE()
(X_train, y_train) = smt.fit_resample(X_train, y_train)
len(y_train)
pass
ssc = StandardScaler()
X_scaled = ssc.fit_transform(X)
X_scaled
import pickle
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()