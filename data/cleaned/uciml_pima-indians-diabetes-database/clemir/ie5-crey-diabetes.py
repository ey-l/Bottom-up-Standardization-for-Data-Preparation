import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head(30)

def replace_0(df, col):
    df1 = df.copy()
    n = df.shape[0]
    m = df[col].mean()
    s = df[col].std()
    for i in range(n):
        if df.loc[i, col] == 0:
            df1.loc[i, col] = np.random.normal(m, s)
    return df1
df = replace_0(df, 'Insulin')
df.head(40)