import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isnull().sum()
sns.countplot(y='Outcome', data=df)
sns.pairplot(df, hue='Outcome')
sns.heatmap(df.corr(), cmap='Accent')
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
X = df[df.columns[:-1]]
y = df['Outcome']