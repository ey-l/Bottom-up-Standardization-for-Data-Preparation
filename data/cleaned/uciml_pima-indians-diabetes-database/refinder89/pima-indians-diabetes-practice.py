import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import platform
if platform.system() == 'Windows':
    path = 'c:/Windows/Fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
elif platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df.info()
df.isnull().sum()
df.isnull()
import missingno as msno
msno.matrix(df)
df.describe()
df.columns
feature_columns = df.columns[:-1].to_list()
feature_columns
cols = feature_columns[1:]
cols
df_null = df[cols].replace(0, np.nan)
df_null = df_null.isnull()
df_null.sum()
df_null.sum().plot.barh()
df_null.mean() * 100
plt.figure(figsize=(15, 5))
sns.heatmap(df_null, cmap='Greys_r')
df['Outcome'].value_counts()
df['Outcome'].value_counts(normalize=True)
df.groupby(['Pregnancies'])['Outcome'].mean()
df.groupby(['Pregnancies'])['Outcome'].agg(['mean', 'count'])
df_po = df.groupby(['Pregnancies'])['Outcome'].agg(['mean', 'count']).reset_index()
df_po
df_po.plot()
(fig, (ax1, ax2)) = plt.subplots(nrows=1, ncols=2)
df_po['mean'].plot(ax=ax1)
df_po['mean'].plot.bar(ax=ax2, rot=0, figsize=(16, 8))
(fig, (ax1, ax2)) = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(18, 8)
sns.countplot(data=df, x='Outcome', ax=ax1)
sns.countplot(data=df, x='Pregnancies', ax=ax2, hue='Outcome')
df['Pregnancies_high'] = df['Pregnancies'] >= 7
df[['Pregnancies', 'Pregnancies_high']].head()
sns.countplot(data=df, x='Pregnancies_high', hue='Outcome')
sns.barplot(data=df, x='Outcome', y='BMI')
sns.barplot(data=df, x='Outcome', y='Glucose')
sns.barplot(data=df, x='Outcome', y='Insulin')
(fig, (ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3)
fig.set_size_inches(18, 8)
sns.barplot(data=df, x='Outcome', y='BMI', ax=ax1)
sns.barplot(data=df, x='Outcome', y='Glucose', ax=ax2)
sns.barplot(data=df, x='Outcome', y='Insulin', ax=ax3)
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Pregnancies', y='Outcome')
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Pregnancies', y='Glucose', hue='Outcome')
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Pregnancies', y='BMI', hue='Outcome')
plt.figure(figsize=(20, 9))
sns.boxplot(data=df, x='Pregnancies', y='Insulin', hue='Outcome')