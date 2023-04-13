import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df.dtypes
df.shape
df.isnull().sum()
df.corr()
from sklearn.feature_selection import SelectKBest, chi2
x = df.iloc[:, :8]
y = df.iloc[:, -1]
bestfeatures = SelectKBest(score_func=chi2, k=8)