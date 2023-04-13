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
pass
pass
df['Outcome'].value_counts()
df['Outcome'].value_counts(normalize=True)
df.groupby(['Pregnancies'])['Outcome'].mean()
df.groupby(['Pregnancies'])['Outcome'].agg(['mean', 'count'])
df_po = df.groupby(['Pregnancies'])['Outcome'].agg(['mean', 'count']).reset_index()
df_po
df_po.plot()
pass
df_po['mean'].plot(ax=ax1)
df_po['mean'].plot.bar(ax=ax2, rot=0, figsize=(16, 8))
pass
fig.set_size_inches(18, 8)
pass
pass
df['Pregnancies_high'] = df['Pregnancies'] >= 7
df[['Pregnancies', 'Pregnancies_high']].head()
pass
pass
pass
pass
pass
fig.set_size_inches(18, 8)
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
pass