import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.columns
_input1.head()
_input1.isnull().sum()
_input1.info()

def missing_imputation(x, stats='mean'):
    if (x.dtypes == 'float64') | (x.dtypes == 'int64'):
        x = x.fillna(x.mean()) if stats == 'mean' else x.fillna(x.median())
    return x
df = _input1.copy()
df.info()
df.describe()
df.head()
df = df.drop(columns=['Name', 'PassengerId', 'Cabin'], axis=1, inplace=False)
df.info()
num = df.select_dtypes(include=['float64']).columns
num
cat = df.select_dtypes(include=['object']).columns
cat

def fillna(col):
    col = col.fillna(col.value_counts().index[0], inplace=False)
    return col
df[cat] = df[cat].apply(lambda col: fillna(col))

def fillna(col):
    col = col.fillna(col.mean(), inplace=False)
    return col
df[num] = df[num].apply(lambda col: fillna(col))
df.isnull().sum()
df = df.dropna(axis=0, subset=['Transported'])
df.Transported.isnull().value_counts()
corr_matrix = df.corr()
plt.figure(figsize=(15, 9))
sns.heatmap(df.corr(), annot=True)
df['Transported'] = df['Transported'].astype(int)
df.info()
(train, test) = train_test_split(df, test_size=0.3, random_state=42)
train.columns
model_eq = 'Transported ~ ' + ' + '.join(train.columns.difference(['Transported']))