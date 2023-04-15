import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isnull().sum()
df.info()
df.describe().T
y = df.Outcome.values
X_ = df.drop(['Outcome'], axis=1)
X = (X_ - np.min(X_)) / (np.max(X_) - np.min(X_)).values
import statsmodels.api as sm
lj = sm.Logit(y, X)