import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.info()
df.describe()
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    df[col] = df[col].replace(0, np.NaN)
    median = int(df[col].median(skipna=True))
    df[col] = df[col].replace(np.NaN, median)
df.head()
df.isnull().sum()
corr = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True)
plt.figure(figsize=(30, 10))
sns.countplot(df['SkinThickness'])
df_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for i in df_columns:
    g = sns.FacetGrid(df, col='Outcome')
    g = g.map(sns.kdeplot, i)
sns.set_style = 'whitegrid'
sns.pairplot(df, hue='Outcome', palette='coolwarm')
plt.figure(figsize=(10, 10))
sns.scatterplot(x='Insulin', y='BMI', hue='Outcome', data=df)
plt.subplot(111)
sns.distplot(df['Age'], bins=10, kde=True)

plt.subplot(121)
sns.distplot(df['Glucose'], bins=10, kde=True)

plt.subplot(131)
sns.distplot(df['BloodPressure'], bins=10, kde=True)

sns.stripplot(x='Outcome', y='Age', data=df)
sns.stripplot(x='Outcome', y='Glucose', data=df)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=11, metric='euclidean', p=2)