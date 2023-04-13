import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe()
df.info()
df.isna().any()
df.notna().any()
df.isna().all()
df.notna().all()
df.notna().sum()
df['Overweight'] = [1 if x > 25 else 0 for x in df.BMI]
df.head()
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
ax.scatter(df.Age, df.Insulin, c=df.Overweight, cmap='viridis')
pass
pass
ax.set_title('Relationship between Age and Insulin')
pass
ax.hist(df.Age, label='Age', bins=10)
pass
pass
bins = [20, 30, 40, 50, 60, 70, 80]
pass
ax.hist(df.Age, label='Age Bins', bins=bins)
pass
pass
pass
ax.bar(df.Outcome, df.Insulin)
pass
pass
pass
ax.bar(df.Age, df.Insulin)
ax.set_xticklabels(df.Age, rotation=45)
fig.savefig('Age.png')
pass
pass
pass
pass
g.fig.suptitle('Age Counts', y=1.04)
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
correlation = df.corr()
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