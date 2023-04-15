import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from scipy.stats import stats
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df1 = df
df.shape
df.head()
df.tail()
df.info()
df.describe().T
len(df[df['Pregnancies'] == 0])
print('Number of Zeros in the columns:')
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols:
    zeros = len(df[df[col] == 0])
    print(col, str(zeros))
for col in cols:
    df[col] = df[col].replace(0, np.NaN)
    mean = int(df[col].mean(skipna=True))
    df[col] = df[col].replace(np.NaN, mean)
df.head()
df.describe().T
df['Outcome'].value_counts()
df.Outcome.value_counts() / df.shape[0] * 100
for col in df.columns:
    print('Skewness of Feature {} is {}'.format(col, df[col].skew(axis=0)))
    sns.displot(df[col], kind='kde')
sns.boxplot(df['Pregnancies'])
sns.boxplot(df['Glucose'])
sns.boxplot(df['BloodPressure'])
sns.boxplot(df['SkinThickness'])
sns.boxplot(df['Insulin'])
sns.boxplot(df['BMI'])
sns.boxplot(df['DiabetesPedigreeFunction'])
sns.boxplot(df['Age'])
for col in df.drop('Outcome', axis=1).columns:
    print(col)
    p_value = stats.normaltest(df[col])[1]
    if p_value > 0.05:
        print('Normality test failed for the feature ', col)
df_corr = df.corr()
(fig, ax) = plt.subplots(figsize=(12, 12))
sns.heatmap(df_corr, annot=True, linewidths=0.5, ax=ax)
print(df_corr['Outcome'].sort_values(ascending=False), '\n')
sns.pairplot(df, corner=True, diag_kind='kde')
X = df.drop('Outcome', axis=1)
y = df['Outcome']