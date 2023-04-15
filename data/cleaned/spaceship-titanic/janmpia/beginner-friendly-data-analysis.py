import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_column', 111)
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
df = data.copy()
df.shape
df.dtypes.value_counts()
df.head()
plt.figure(figsize=(20, 10))
sns.heatmap(df.isna(), cbar=False)
(df.isna().sum() / df.shape[0]).sort_values(ascending=False)
sns.clustermap(df.corr(method='kendall'))
plt.figure(figsize=(10, 10))
plt.title('Transported rate')
(df['Transported'].value_counts(normalize=True) * 100).plot.pie(labels=['Transported', 'Not Transported'], autopct='%1.1f%%')
plt.figure()
sns.displot(df['Age'], kde=True)

for col in df.select_dtypes('object'):
    print(f'{col:-<20} {df[col].unique()}')
for col in df.select_dtypes('object'):
    plt.figure()
    if col not in ['PassengerId', 'Cabin', 'Name']:
        df[col].value_counts().plot.pie()