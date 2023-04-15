import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import random
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (22, 5)
plt.rcParams['figure.dpi'] = 100
from sklearn.metrics import r2_score
from urllib.request import urlretrieve
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
urlretrieve(medical_charges_url, 'medical.csv')
medical_df = pd.read_csv('medical.csv')
medical_df.head()
medical_df.shape
medical_df.describe()
medical_df.info()
sns.catplot(data=medical_df, x='sex', y='charges', kind='swarm', hue='smoker', aspect=2, height=8)
plt.axhline(medical_df['charges'].mean(), linestyle='--', lw=2, zorder=1, color='black')
plt.annotate(f' Average Medical Charges ($)', (0.4, medical_df['charges'].mean() + 900), fontsize=14, color='red')
plt.title('Insurance Charges Gender Wise', fontsize=20)

sns.catplot(data=medical_df, x='region', y='charges', kind='box', aspect=2, height=8)
plt.axhline(medical_df['charges'].mean(), linestyle='--', lw=4, zorder=1, color='red')
plt.title('Medical Charges Region Wise', fontsize=18)
plt.xlabel('Region')
plt.ylabel('Medical Charges ($)')

sns.barplot(data=medical_df, x='smoker', y='charges', hue='smoker', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.axhline(medical_df['charges'].mean(), linestyle='--', lw=2, zorder=1, color='red')
plt.title('Insurance Charges smoker Wise', fontsize=18)
plt.ylabel('Medical Charges ($)')

(fig, ax) = plt.subplots()
(N, bins, patches) = ax.hist(np.array(medical_df.charges), edgecolor='white', color='lightgray', linewidth=5, alpha=0.7)
for i in range(0, 1):
    patches[i].set_facecolor('orange')
    plt.title('Medical Charges Histogram', fontsize=18)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('Medical Charges ($)')
    plt.ylabel('Count')
    plt.axvline(medical_df['charges'].mean(), linestyle='--', lw=2, zorder=1, color='blue')
    plt.annotate(f' Average Medical Charges ($)', (13500, 500), fontsize=14, color='red')

sns.scatterplot(y=medical_df['charges'], x=medical_df['age'], hue=medical_df['smoker'], alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.axhline(medical_df['charges'].mean(), linestyle='--', lw=2, zorder=1, color='black')
plt.annotate(f'Average Medical Charges ($)', (45, 13900), fontsize=14, color='red')
plt.title('Age Wise Medical Charges Distribution', fontsize=18)
plt.ylabel('Medical Charges ($)')

sns.scatterplot(y=medical_df['bmi'], x=medical_df['charges'], hue=medical_df['smoker'], alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.axvline(medical_df['charges'].mean(), linestyle='--', lw=2, zorder=1, color='black')
plt.annotate(f'Average Medical Charges ($)', (13400, 52), fontsize=14, color='Red')
plt.title('BMI & Medical Charges relation', fontsize=18)
plt.xlabel('Medical Charges ($)')
plt.ylabel('BMI')

sns.relplot(data=medical_df, x='children', y='charges', hue='smoker', aspect=2, height=8)
plt.axhline(medical_df['charges'].mean(), linestyle='--', lw=2, zorder=1, color='black')
plt.annotate(f'Average Medical Charges ($)', (4.05, 13900), fontsize=12, color='red')
plt.title('Children & Medical Charges relation', fontsize=18)
plt.ylabel('Medical Charges ($)')
plt.xlabel('Number of Children')

sns.boxplot(x=medical_df['charges'])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.axvline(medical_df['charges'].mean(), linestyle='--', lw=4, zorder=1, color='red')
plt.annotate(f'Average Medical Charges ($)', (13800, 0.47), fontsize=15, color='red')
plt.title('Outliers in the data', fontsize=18)
plt.xlabel('Medical Charges ($)')

ax = sns.histplot(medical_df['charges'], kde=True, color='lightgray')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax.lines[0].set_color('red')
plt.axvline(medical_df['charges'].mean(), linestyle='--', lw=3, zorder=1, color='blue')
plt.annotate(f'Average Medical Charges ($)', (13700, 200), fontsize=15, color='blue')
plt.title('Detecting Outliers Using Histogram', fontsize=18)
plt.xlabel('Medical Charges ($)')

data = sorted(medical_df['charges'].values)
(data_mean, data_std) = (np.mean(data), np.std(data))
cut_off = data_std * 3
(lower, upper) = (data_mean - cut_off, data_mean + cut_off)
print('Cut Off =', round(cut_off, 3))
print('Lower =', round(lower, 3))
print('Upper =', round(upper, 3))
ax = sns.histplot(medical_df['charges'], kde=True, color='lightgray')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax.lines[0].set_color('red')
plt.axvline(data_mean, linestyle='--', lw=2, zorder=1, color='orange')
plt.annotate(f'Average', (data_mean + 500, 175), fontsize=15, color='blue')
plt.axvline(upper, linestyle='--', lw=2, zorder=1, color='orange')
plt.annotate(f'Upper', (upper + 500, 175), fontsize=15, color='blue')
plt.axvline(cut_off, linestyle='--', lw=2, zorder=1, color='orange')
plt.annotate(f'Cut Off', (cut_off + 500, 175), fontsize=15, color='blue')
plt.title('Detecting Outliers', fontsize=18)
plt.xlabel('Medical Charges ($)')

medical_df = medical_df[medical_df['charges'] < upper]
medical_df = medical_df[medical_df['charges'] > lower]
print('The shape of our dataframe after the Outlier Removal is', medical_df.shape)
df = medical_df.copy()
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
cat_cols = df.select_dtypes(include='object').columns