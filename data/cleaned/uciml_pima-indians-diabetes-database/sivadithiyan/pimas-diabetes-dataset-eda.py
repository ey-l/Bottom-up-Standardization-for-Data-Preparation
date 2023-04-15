import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotly
import plotly.express as px
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.tail()
df.dtypes
df.info()
print(df.isnull().sum())
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange']
for (i, col) in enumerate(df.columns):
    plt.figure(figsize=(4, 4))
    plt.hist(df[col], bins=50, color=colors[i])
    plt.title('Histogram of ' + col)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend([col])

sns.set(style='ticks')
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange']
for (i, col) in enumerate(df.columns):
    plt.figure(figsize=(5, 5))
    sns.histplot(data=df, x=col, kde=True, color=colors[i], edgecolor='black')
    plt.title('Histogram of ' + col)
    plt.xlabel(col)
    plt.ylabel('Frequency')

outcome_counts = df['Outcome'].value_counts()
plt.bar(outcome_counts.index, outcome_counts.values)
plt.title('Count of Outcome Values')
plt.xlabel('Outcome')
plt.ylabel('Count')

sns.set(style='darkgrid')
sns.countplot(x='Outcome', data=df)
plt.title('Count of Outcome Values')
plt.xlabel('Outcome')
plt.ylabel('Count')

import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
sns.set_palette('husl')
for col in df.columns[:-1]:
    plt.figure(figsize=(8, 5))
    sns.kdeplot(df.loc[df['Outcome'] == 0, col], shade=True, label='Outcome 0')
    sns.kdeplot(df.loc[df['Outcome'] == 1, col], shade=True, label='Outcome 1')
    plt.xlabel(col)
    plt.legend()

plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='magma')
plt.title('Correlation Matrix - Pima Indians Diabetes dataset')

plt.figure(figsize=(10, 8))
fig = px.parallel_coordinates(df, color='Outcome', dimensions=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
fig.show()
sns.pairplot(df, hue='Outcome')
sns.set_style('darkgrid')
custom_palette = ['blue', 'red']
sns.set_palette(custom_palette)
sns.pairplot(df, hue='Outcome')
