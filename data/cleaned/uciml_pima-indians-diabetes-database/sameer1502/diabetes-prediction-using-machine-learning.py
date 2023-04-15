import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
sns.countplot(data=df, x='Outcome')
df.shape
df.describe()
df.isnull().sum()
df.corr()
sns.heatmap(df.corr(), annot=True)
for i in df.columns:
    print(i, len(df[df[i] == 0]))
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
df.isnull().sum()
sns.pairplot(df)
df.isnull().sum()
df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace=True)
df['Insulin'].fillna(df['Insulin'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].mean(), inplace=True)
df.isnull().sum()
df['Outcome'].value_counts()
from sklearn.model_selection import train_test_split
X = df.drop('Outcome', axis=1)
X.head()
y = df['Outcome']
y.head()
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=10)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()