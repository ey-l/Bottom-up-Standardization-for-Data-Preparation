import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(df.head())
df.info()
df.shape
coeff = df.corr()
plt.figure(figsize=(22, 7))
plt.title('correlation heatmap')
sns.heatmap(coeff, annot=True)

pp.ProfileReport(df)