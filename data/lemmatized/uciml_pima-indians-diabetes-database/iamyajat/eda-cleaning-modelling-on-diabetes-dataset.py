import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import missingno as msno
from pylab import rcParams
rcParams['figure.figsize'] = (15, 10)
import random
pallete = ['Accent_r', 'Blues', 'BrBG', 'BrBG_r', 'BuPu', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'OrRd', 'Oranges', 'Paired', 'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdGy_r', 'RdPu', 'Reds', 'autumn', 'cool', 'coolwarm', 'flag', 'flare', 'gist_rainbow', 'hot', 'magma', 'mako', 'plasma', 'prism', 'rainbow', 'rocket', 'seismic', 'spring', 'summer', 'terrain', 'turbo', 'twilight']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
df['Glucose'] = df['Glucose'].apply(lambda x: np.nan if x == 0 else x)
df['BloodPressure'] = df['BloodPressure'].apply(lambda x: np.nan if x == 0 else x)
df['SkinThickness'] = df['SkinThickness'].apply(lambda x: np.nan if x == 0 else x)
df['Insulin'] = df['Insulin'].apply(lambda x: np.nan if x == 0 else x)
df['BMI'] = df['BMI'].apply(lambda x: np.nan if x == 0 else x)
df.isnull().sum()
px.pie(df, names='Outcome')
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
axs = axs.flatten()
for i in range(len(df.columns) - 1):
    pass
pass
df.isnull().sum()
msno.bar(df)
msno.matrix(df, figsize=(20, 35))
msno.heatmap(df, cmap=random.choice(pallete))
msno.dendrogram(df)
df.isnull().sum() / len(df) * 100
df.drop(columns=['Insulin'], inplace=True)
df.describe()
df.skew()
df['BMI'].replace(to_replace=np.nan, value=df['BMI'].median(), inplace=True)
df['Pregnancies'].replace(to_replace=np.nan, value=df['Pregnancies'].median(), inplace=True)
df['Glucose'].replace(to_replace=np.nan, value=df['Glucose'].mean(), inplace=True)
df['BloodPressure'].replace(to_replace=np.nan, value=df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].replace(to_replace=np.nan, value=df['SkinThickness'].mean(), inplace=True)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
df_out = df[~((df < Q1 - 1.5 * IQR) | (df > Q3 + 1.5 * IQR)).any(axis=1)]
print(f'Before: {df.shape}, After: {df_out.shape}')
for col in df.columns[:-1]:
    up_out = df[col].quantile(0.9)
    low_out = df[col].quantile(0.1)
    med = df[col].median()
    df[col] = np.where(df[col] > up_out, med, df[col])
    df[col] = np.where(df[col] < low_out, med, df[col])
df.describe()
X = df_out[df_out.columns[:-1]]
y = df_out['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)