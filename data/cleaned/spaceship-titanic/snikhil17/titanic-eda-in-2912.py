import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
import statsmodels.api as sm
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
sample_submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
df.head()
df.info()
df.shape


missing_val_df = (df.isnull().sum() / df.shape[0] * 100).to_frame().reset_index().rename({'index': 'columns', 0: 'Missing Values percentage'}, axis=1)
plt.figure(figsize=(10, 8))
sns.barplot(data=missing_val_df, x='Missing Values percentage', y='columns', palette='dark')
plt.title('Missing Values percentage Training Data', fontsize=20, fontweight='bold', color='#bd0b0b')
plt.xlabel(' ')
plt.ylabel(' ')
plt.xticks(fontsize=12, fontweight='bold', color='maroon')
plt.yticks(fontsize=12, fontweight='bold', color='maroon')
missing_val_df_test = (df_test.isnull().sum() / df_test.shape[0] * 100).to_frame().reset_index().rename({'index': 'columns', 0: 'Missing Values percentage'}, axis=1)
plt.figure(figsize=(10, 8))
sns.barplot(data=missing_val_df_test, x='Missing Values percentage', y='columns', palette='dark')
plt.title('Missing Values percentage Testing Data', fontsize=20, fontweight='bold', color='#bd0b0b')
plt.xlabel(' ')
plt.ylabel(' ')
plt.xticks(fontsize=12, fontweight='bold', color='maroon')
plt.yticks(fontsize=12, fontweight='bold', color='maroon')

plt.figure(figsize=(10, 8))
sns.countplot(data=df, x='Transported', palette='dark')
plt.title('Transported', fontsize=20, fontweight='bold', color='#bd0b0b')
plt.xlabel(' ')
plt.ylabel(' ')
plt.xticks(fontsize=12, fontweight='bold', color='navy')
plt.yticks(fontsize=12, fontweight='bold', color='navy')
df.info()
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
for col in cat_cols:
    print(f'Number of Rows: {df.shape[0]}')
    print(col)
    print(f'First 5 Unique Values: {df[col].unique()[:5]}')
    print(f'Number of Unique Values: {df[col].nunique()}')
    print(f'Value Counts: \n{df[col].value_counts(dropna=False).head(5)}')
    print('=' * 50)
df[['CryoSleep', 'Destination', 'VIP']].groupby(['Destination', 'VIP'])[['CryoSleep']].agg(['count', 'mean', 'median', pd.Series.mode])
df['isTrain'] = True
df_test['isTrain'] = False
tt = pd.concat([df, df_test]).reset_index(drop=True).copy()
tt.head()
'Mode for Destination and VIP '
tt['Destination'] = tt['Destination'].fillna(tt['Destination'].mode()[0])
tt['VIP'] = tt['VIP'].fillna(tt['VIP'].mode()[0])
'For CyroSleep, Cabin, HomePlanet: Mode after Grouping by Destination and VIP'
tt['CryoSleep'] = tt.groupby(['Destination', 'VIP'])['CryoSleep'].transform(lambda x: x.fillna(x.mode()[0]))
tt['Cabin'] = tt.groupby(['CryoSleep', 'VIP'])['Cabin'].transform(lambda x: x.fillna(x.mode()[0]))
tt['HomePlanet'] = tt.groupby(['CryoSleep', 'VIP'])['HomePlanet'].transform(lambda x: x.fillna(x.mode()[0]))
'For missing Names: Used keyword- Missing'
tt['Name'] = tt['Name'].fillna('MISSING')
'Created List of columns according to the data-types'
useful_cols = [col for col in tt.columns if col not in ['PassengerId', 'Name', 'isTrain', 'Transported']]
cat_cols = [col for col in useful_cols if tt[col].dtype == 'object']
bool_cols = ['VIP', 'CryoSleep']
col_num = [col for col in useful_cols if col not in cat_cols + bool_cols]
'Preprocessing for Using Iterative Imputer'
comb_df = tt.copy()
le = preprocessing.LabelEncoder()
for i in comb_df[cat_cols].columns:
    comb_df[i] = le.fit_transform(comb_df[i])
'Imputing missing values of Float-type data using Iterative Imputer'
it_imputer = IterativeImputer(max_iter=1000)
iterimp = it_imputer.fit_transform(comb_df[useful_cols])
imp_df = pd.DataFrame(iterimp, columns=comb_df[useful_cols].columns)
for col in comb_df[col_num].columns:
    tt[col] = imp_df[col]
tt.head()
'Taking out Training and Testing set from combined Dataset.'
train = tt[tt.isTrain == True].drop('isTrain', axis=1)
test = tt[tt.isTrain != True].drop(['isTrain', 'Transported'], axis=1)
'Checking missing values'
print(f'Missing values in Training Set: {train.isnull().sum().sum()} \nMissing values in Test Set: {test.isnull().sum().sum()}')
train.describe().style.background_gradient(cmap='Pastel1')
'Univariate Analysis on Categorical Columns'
'Pie Chart of Pregnencies and Outcome'
for i in ['HomePlanet', 'Destination']:
    train.groupby(i)[i].count().plot.pie(y=i, figsize=(10, 10), cmap='copper')

    plt.tight_layout()
'Historgram of categorical variables'
plt.figure(figsize=(16, 6))
for (i, col) in enumerate(['HomePlanet', 'Destination']):
    plt.subplot(1, 2, i + 1)
    sns.histplot(y=col, data=train, color='#0D5160', alpha=1)
    plt.xticks(rotation=0)
    plt.title(col, fontsize=15, fontweight='bold', color='#bd0b0b')
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.xticks(fontsize=12, fontweight='bold', color='brown')
    plt.yticks(fontsize=12, fontweight='bold', color='brown')
    plt.tight_layout()
'Univariate Analysis of Boolean Cols'
plt.figure(figsize=(16, 10))
for (i, col) in enumerate(bool_cols + ['Transported']):
    plt.subplot(2, 2, i + 1)
    sns.countplot(y=col, data=train, color='#5A045F')
    plt.xticks(rotation=0)
    plt.title(col, fontsize=15, fontweight='bold', color='#bd0b0b')
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.xticks(fontsize=12, fontweight='bold', color='brown')
    plt.yticks(fontsize=12, fontweight='bold', color='brown')
    plt.tight_layout()
'Univariate Analysis of Numerical Cols'
plt.figure(figsize=(16, 10))
for (i, col) in enumerate(col_num):
    plt.subplot(3, 2, i + 1)
    sns.kdeplot(x=col, data=train, color='#830D0D', fill='#830D0D', alpha=1)
    plt.xticks(rotation=0)
    plt.title(col, fontsize=15, fontweight='bold', color='#bd0b0b')
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.xticks(fontsize=12, fontweight='bold', color='brown')
    plt.yticks(fontsize=12, fontweight='bold', color='brown')
    plt.tight_layout()
'Bivariate Analysis of categorical variables'
plt.figure(figsize=(16, 7))
for (i, col) in enumerate(['HomePlanet', 'Destination']):
    plt.subplot(1, 2, i + 1)
    sns.countplot(y=col, data=train, palette='dark', hue='Transported')
    plt.xticks(rotation=0)
    plt.title(f'{col} with hue: Transported', fontsize=15, fontweight='bold', color='#bd0b0b')
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.xticks(fontsize=12, fontweight='bold', color='brown')
    plt.yticks(fontsize=12, fontweight='bold', color='brown')
    plt.tight_layout()
'Bi-variate Analysis of Boolean Cols'
plt.figure(figsize=(16, 10))
for (i, col) in enumerate(bool_cols):
    plt.subplot(2, 2, i + 1)
    sns.countplot(y=col, data=train, hue='Transported', palette='dark')
    plt.xticks(rotation=0)
    plt.title(f'{col} with hue: Transported', fontsize=15, fontweight='bold', color='#bd0b0b')
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.xticks(fontsize=12, fontweight='bold', color='brown')
    plt.yticks(fontsize=12, fontweight='bold', color='brown')
    plt.tight_layout()
'Bivariate Analysis of Numerical Cols'
plt.figure(figsize=(16, 10))
for (i, col) in enumerate(col_num):
    plt.subplot(3, 2, i + 1)
    sns.kdeplot(x=col, data=train, fill='Transported', hue='Transported', palette='dark', alpha=0.7)
    plt.xticks(rotation=0)
    plt.title(col, fontsize=15, fontweight='bold', color='#bd0b0b')
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.xticks(fontsize=12, fontweight='bold', color='brown')
    plt.yticks(fontsize=12, fontweight='bold', color='brown')
    plt.tight_layout()
plt.figure(figsize=(25, 50))
for (i, col) in enumerate(train[col_num].columns):
    plt.subplot(5, 2, i + 1)
    sns.boxenplot(x='Transported', y=col, data=train, palette='dark')
    plt.xticks(rotation=0)
    plt.title(f'{col} vs Transported', fontsize=20, fontweight='bold', color='#bd0b0b')
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.xticks(fontsize=15, fontweight='bold', color='brown')
    plt.yticks(fontsize=15, fontweight='bold', color='brown')
    plt.tight_layout()

'Correlation Matrix'
corr_matrix = train.corr()
matrix = np.tril(corr_matrix)
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix.T, mask=matrix, square=True, cmap='RdBu', annot=True, alpha=1)
sns.pairplot(train[col_num + ['Transported']], hue='Transported', palette='dark')