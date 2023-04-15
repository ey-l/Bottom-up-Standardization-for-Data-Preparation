import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config
set_config(display='diagram')
import warnings
warnings.filterwarnings('ignore')
from matplotlib import rcParams
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(data.shape)
data.head()
data.isnull().sum()
from sklearn.model_selection import train_test_split
(data, test) = train_test_split(data, test_size=0.1, random_state=42)
data.shape
data.info()
data.describe()
data.columns
plt.figure(figsize=(12, 35))
for i in enumerate(data.columns[:-1]):
    plt.subplot(8, 2, 2 * i[0] + 1)
    data[i[1]].hist(grid=False, xlabelsize=12, lw=1.5, ylabelsize=14, edgecolor='black')
    plt.title(f'Distribution of {i[1]}', fontsize=16)
    plt.xlabel(f'{i[1]}', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout(h_pad=5, w_pad=5)
    plt.subplot(8, 2, 2 * i[0] + 2)
    sns.boxplot(data[i[1]])
    plt.title(f'{i[1]} Boxplot', fontsize=16)
    plt.xlabel(f'{i[1]}', fontsize=14)
    plt.tight_layout(h_pad=5, w_pad=5)
for col in data.columns[:-1]:
    print(f'Statistical Maximum for {col}: {data[col].quantile(0.75) + 1.5 * (data[col].quantile(0.75) - data[col].quantile(0.25))}')
    print(f'Statistical Minimum for {col}: {data[col].quantile(0.25) - 1.5 * (data[col].quantile(0.75) - data[col].quantile(0.25))}')
    print()
for (idx, row) in data.iterrows():
    if data.loc[idx, 'Pregnancies'] > 13:
        data.loc[idx, 'Pregnancies'] = 13
for (idx, row) in data.iterrows():
    if data.loc[idx, 'Glucose'] > 200:
        data.loc[idx, 'Glucose'] = 200
    if data.loc[idx, 'Glucose'] < 40:
        data.loc[idx, 'Glucose'] = 40
for (idx, row) in data.iterrows():
    if data.loc[idx, 'BloodPressure'] > 107:
        data.loc[idx, 'BloodPressure'] = 107
    if data.loc[idx, 'BloodPressure'] < 35:
        data.loc[idx, 'BloodPressure'] = 35
for (idx, row) in data.iterrows():
    if data.loc[idx, 'Insulin'] > 323:
        data.loc[idx, 'Insulin'] = 323
for (idx, row) in data.iterrows():
    if data.loc[idx, 'BMI'] > 50:
        data.loc[idx, 'BMI'] = 50
    if data.loc[idx, 'BMI'] < 14:
        data.loc[idx, 'BMI'] = 14
for (idx, row) in data.iterrows():
    if data.loc[idx, 'DiabetesPedigreeFunction'] > 1.19:
        data.loc[idx, 'DiabetesPedigreeFunction'] = 1.19
for (idx, row) in data.iterrows():
    if data.loc[idx, 'Age'] > 64:
        data.loc[idx, 'Age'] = 64
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
tempData = data.copy()
tempData = tempData.drop('Outcome', axis=1)
tempData = sc.fit_transform(tempData)
temp_df = pd.DataFrame(tempData, columns=data.columns[:-1])
plt.figure(figsize=(12, 35))
for i in enumerate(data.columns[:-1]):
    plt.subplot(8, 2, 2 * i[0] + 1)
    data[i[1]].hist(grid=False, xlabelsize=12, lw=1.5, ylabelsize=14, edgecolor='black')
    plt.title(f'Distribution of {i[1]}', fontsize=16, pad=30)
    plt.xlabel(f'{i[1]}', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout(h_pad=5, w_pad=5)
    plt.subplot(8, 2, int(f'{2 * i[0] + 2}'))
    stats.probplot(temp_df[i[1]], dist='norm', plot=plt)
    plt.title(f'Normal QQ Plot of {i[1]}', fontsize=16, pad=30)
    plt.xlabel('Theoretical Quantiles (Standard Normal Distribution)', fontsize=14)
    plt.ylabel(f'Scaled {i[1]}', fontsize=14)
plt.rcParams.update({'font.size': 12})
(fig, ax) = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm', ax=ax, lw=0.2, edgecolor='white')
from sklearn.ensemble import ExtraTreesClassifier
fig = plt.figure(figsize=(14, 12))
model = ExtraTreesClassifier()