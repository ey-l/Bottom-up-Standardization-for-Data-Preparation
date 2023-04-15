import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(df)
df.sort_values(by=['Pregnancies'])
cols = df[['BloodPressure', 'Pregnancies']]
print(cols)
rows = df.iloc[[1, 3, 7]]
print(rows)
rowscol = df.iloc[[3, 4], [0, 1]]
print(rowscol)
df.mean(axis=0)
df.fillna(0)
df.dropna()
df.dropna(axis=1)