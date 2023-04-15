import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
sns.set(style='darkgrid')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.isnull().values.any()
df.head()
df.shape
df.columns
df.info()
df.describe()
pd.pivot_table(df, index=['Outcome'], aggfunc=[np.mean])
pd.pivot_table(df, index=['Outcome'], aggfunc=[np.std])
colnames = df.columns
(fig, ax) = plt.subplots(nrows=2, ncols=4, figsize=(15, 12))
for i in range(4):
    x = colnames[i]
    ax[0, i].boxplot(df[str(x)], labels=[str(x)])
    ax[0, i].set_ylabel(str(x))
    ax[0, i].set_title(str(x) + '\nBoxplot')
for i in range(4, 8):
    x = colnames[i]
    ax[1, i - 4].boxplot(df[str(x)], labels=[str(x)])
    ax[1, i - 4].set_ylabel(str(x))
    ax[1, i - 4].set_title(str(x) + '\nBoxplot')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.4)
(fig, ax) = plt.subplots(nrows=2, ncols=4, figsize=(15, 12), sharey=True)
for i in range(4):
    x = colnames[i]
    ax[0, i].hist(df[str(x)], color='red', bins=20)
    ax[0, i].set_ylabel(str(x))
    ax[0, i].set_title(str(x) + '\nHistogram')
for i in range(4, 8):
    x = colnames[i]
    ax[1, i - 4].hist(df[str(x)], color='red', bins=20)
    ax[1, i - 4].set_ylabel(str(x))
    ax[1, i - 4].set_title(str(x) + '\nHistogram')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
(fig, ax) = plt.subplots(nrows=2, ncols=4, figsize=(15, 12))
for i in range(4):
    x = colnames[i]
    ax[0, i].violinplot(df[str(x)])
    ax[0, i].set_ylabel(str(x))
    ax[0, i].set_title(str(x) + '\nViolinplot')
for i in range(4, 8):
    x = colnames[i]
    ax[1, i - 4].violinplot(df[str(x)])
    ax[1, i - 4].set_ylabel(str(x))
    ax[1, i - 4].set_title(str(x) + '\nViolinplot')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.4)
(fig, ax) = plt.subplots(2, 4, figsize=(15, 12))
for i in range(4):
    x = colnames[i]
    sns.barplot(data=df, y=str(x), x='Outcome', ax=ax[0, i])
    ax[0, i].set_title(str(x) + '\nDistribution by Outcome')
for i in range(4, 8):
    x = colnames[i]
    sns.barplot(data=df, y=str(x), x='Outcome', ax=ax[1, i - 4])
    ax[1, i - 4].set_title(str(x) + '\nDistribution by Outcome')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.4)
(fig, ax) = plt.subplots(2, 4, figsize=(15, 12))
for i in range(4):
    x = colnames[i]
    sns.boxplot(data=df, y=str(x), x='Outcome', ax=ax[0, i])
    ax[0, i].set_title(str(x) + '\nDistribution by Outcome')
for i in range(4, 8):
    x = colnames[i]
    sns.boxplot(data=df, y=str(x), x='Outcome', ax=ax[1, i - 4])
    ax[1, i - 4].set_title(str(x) + '\nDistribution by Outcome')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.4)
(fig, ax) = plt.subplots(2, 4, figsize=(15, 12))
for i in range(4):
    x = colnames[i]
    sns.violinplot(data=df, y=str(x), x='Outcome', ax=ax[0, i])
    ax[0, i].set_title(str(x) + '\nDistribution by Outcome')
for i in range(4, 8):
    x = colnames[i]
    sns.violinplot(data=df, y=str(x), x='Outcome', ax=ax[1, i - 4])
    ax[1, i - 4].set_title(str(x) + '\nDistribution by Outcome')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.4)
(fig, ax) = plt.subplots(2, 4, figsize=(15, 12))
for i in range(4):
    x = colnames[i]
    sns.stripplot(data=df, y=str(x), x='Outcome', ax=ax[0, i])
    ax[0, i].set_title(str(x) + '\nDistribution by Outcome')
for i in range(4, 8):
    x = colnames[i]
    sns.stripplot(data=df, y=str(x), x='Outcome', ax=ax[1, i - 4])
    ax[1, i - 4].set_title(str(x) + '\nDistribution by Outcome')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.4)
(fig, ax) = plt.subplots(2, 4, figsize=(15, 12))
for i in range(4):
    x = colnames[i]
    sns.kdeplot(data=df, x=str(x), hue='Outcome', shade=True, ax=ax[0, i])
    ax[0, i].set_title(str(x) + '\nDistribution by Outcome')
for i in range(4, 8):
    x = colnames[i]
    sns.kdeplot(data=df, x=str(x), hue='Outcome', shade=True, ax=ax[1, i - 4])
    ax[1, i - 4].set_title(str(x) + '\nDistribution by Outcome')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.4)
plt.figure(figsize=(12, 10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

sns.pairplot(data=df, hue='Outcome')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
sns.set(style='darkgrid')
import warnings
warnings.filterwarnings('ignore')
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x = pd.DataFrame(x_scaled, columns=x.columns)
from sklearn.model_selection import train_test_split
(itrain_x, test_x, itrain_y, test_y) = train_test_split(x, y, random_state=56, stratify=y, test_size=0.1)
(train_x, valid_x, train_y, valid_y) = train_test_split(itrain_x, itrain_y, random_state=56, stratify=itrain_y, test_size=1 / 9)
print(train_x.shape[0] / x.shape[0], valid_x.shape[0] / x.shape[0], test_x.shape[0] / x.shape[0])
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
clf = KNN(n_neighbors=10)