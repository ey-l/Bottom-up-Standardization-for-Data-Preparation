import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

sns.set()
np.random.seed(42)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.shape
df.info()
df.head()
df.tail()
df.describe()
(fig, ax) = plt.subplots(3, 3, figsize=(20, 20))
features = list(df.columns)
(i, j) = (0, 0)
for feature in features:
    sns.histplot(x=feature, data=df, ax=ax[i][j])
    i += 1
    if i >= 3:
        i = 0
        j += 1

df_copy = df.copy()
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_copy[cols] = df_copy[cols].replace({0: np.nan})
nans = df_copy.isna().sum().sort_values(ascending=False)
nans_pct = 100 * nans / df_copy.shape[0]
missing_values = pd.concat([nans, nans_pct], axis=1).rename({0: 'NumOfNaN', 1: 'PctOfNaN'}, axis='columns')
missing_values.head()
t = df_copy.groupby('Outcome').describe().T
t.loc['Glucose', :]
plt.figure(figsize=(6, 4))
sns.histplot(hue='Outcome', x='Glucose', data=df_copy)

missing_values.loc[['Glucose']]
df_copy[df_copy['Glucose'].isna()]
t.loc['Glucose', :]
g_median = t.loc['Glucose', '50%']
g_median
filt = df_copy['Glucose'].isna()
for i in range(2):
    to_fill = filt & (df_copy['Outcome'] == i)
    df_copy.loc[to_fill, 'Glucose'] = g_median[i]
df_copy['Glucose'].isna().any()
missing_values.loc[['BMI']]
filt = df_copy['BMI'].isna()
df_copy[filt]
t.loc['BMI', :]
bmi_m = df_copy.groupby('Outcome')['BMI'].median()
bmi_m
for i in range(2):
    to_fill = filt & (df_copy['Outcome'] == i)
    df_copy.loc[to_fill, 'BMI'] = bmi_m[i]
df_copy['BMI'].isna().any()
missing_values.loc[['BloodPressure']]
t.loc['BloodPressure', :]
bp_nan = df_copy['BloodPressure'].isna()
bp_nan_0 = bp_nan & (df['Outcome'] == 0)
bp_nan_1 = bp_nan & (df['Outcome'] == 1)
df_copy[bp_nan_0].head()
df_copy[bp_nan_1].head()
rand_bp_0 = np.round(np.random.normal(t.loc['BloodPressure', 'mean'][0], t.loc['BloodPressure', 'std'][0] // 3, size=bp_nan_0.sum()))
rand_bp_1 = np.round(np.random.normal(t.loc['BloodPressure', 'mean'][1], t.loc['BloodPressure', 'std'][1] // 3, size=bp_nan_1.sum()))
rand_bp_0
rand_bp_1
df_copy.loc[bp_nan_0, 'BloodPressure'] = rand_bp_0
df_copy.loc[bp_nan_1, 'BloodPressure'] = rand_bp_1
(fig, ax) = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)
sns.histplot(df_copy['BloodPressure'], ax=ax[0])
sns.histplot(df['BloodPressure'], ax=ax[1])

df['BloodPressure'].isna().any()
missing_values.loc[['Insulin']]
t.loc['Insulin', :]
insu_nan = df_copy['Insulin'].isna()
insu_nan_0 = insu_nan & (df['Outcome'] == 0)
insu_nan_1 = insu_nan & (df['Outcome'] == 1)
df_copy[insu_nan_0].head()
df_copy[insu_nan_1].head()
rand_insu_0 = np.round(np.random.normal(t.loc['Insulin', 'mean'][0], t.loc['Insulin', 'std'][0] // 4, size=insu_nan_0.sum()))
rand_insu_1 = np.round(np.random.normal(t.loc['Insulin', 'mean'][1], t.loc['Insulin', 'std'][1] // 4, size=insu_nan_1.sum()))
rand_insu_0[:10]
rand_insu_1[:10]
df_copy.loc[insu_nan_0, 'Insulin'] = rand_insu_0
df_copy.loc[insu_nan_1, 'Insulin'] = rand_insu_1
(fig, ax) = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)
sns.histplot(df_copy['Insulin'], ax=ax[0])
sns.histplot(df['Insulin'], ax=ax[1])

df_copy['Insulin'].isna().any()
plt.figure(figsize=(6, 4))
sns.histplot(hue='Outcome', x='SkinThickness', data=df_copy)

missing_values.loc[['SkinThickness']]
t.loc['SkinThickness', :]
sk_nan = df_copy['SkinThickness'].isna()
sk_nan_0 = sk_nan & (df['Outcome'] == 0)
sk_nan_1 = sk_nan & (df['Outcome'] == 1)
df_copy[sk_nan_0].head()
df_copy[sk_nan_1].head()
rand_sk_0 = np.round(np.random.normal(t.loc['SkinThickness', 'mean'][0], t.loc['SkinThickness', 'std'][0] // 4, size=sk_nan_0.sum()))
rand_sk_1 = np.round(np.random.normal(t.loc['SkinThickness', 'mean'][1], t.loc['SkinThickness', 'std'][1] // 4, size=sk_nan_1.sum()))
rand_sk_0[:10]
rand_sk_1[:10]
df_copy.loc[sk_nan_0, 'SkinThickness'] = rand_sk_0
df_copy.loc[sk_nan_1, 'SkinThickness'] = rand_sk_1
(fig, ax) = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)
sns.histplot(df_copy['SkinThickness'], ax=ax[0])
sns.histplot(df['SkinThickness'], ax=ax[1], bins=30)

df_copy['SkinThickness'].isna().any()
df_copy['Outcome'].value_counts()
survived = df_copy['Outcome'].replace({0: 'No', 1: 'Yes'}).value_counts()
plt.figure(dpi=120)
plt.pie(survived.values, labels=survived.index, startangle=90, autopct='%1.2f%%', labeldistance=None, textprops={'fontsize': 12}, shadow=True, explode=[0, 0.12])
plt.legend()
plt.title('Has Diabetes', fontsize=14)

t = df_copy.groupby('Outcome').describe().T
t.loc['Pregnancies', :]
plt.figure(figsize=(6, 6))
sns.boxplot(x='Outcome', y='Pregnancies', data=df_copy)
plt.yticks(list(range(0, 18, 2)))

t.loc['Glucose', :]
plt.figure(figsize=(6, 6))
sns.boxplot(x='Outcome', y='Glucose', data=df_copy)

t.loc['BloodPressure', :]
plt.figure(figsize=(6, 6))
sns.boxplot(x='Outcome', y='BloodPressure', data=df_copy)

t.loc['SkinThickness', :]
plt.figure(figsize=(6, 6))
sns.boxplot(x='Outcome', y='SkinThickness', data=df_copy)

t.loc['Insulin', :]
plt.figure(figsize=(6, 6))
sns.boxplot(x='Outcome', y='Insulin', data=df_copy)

t.loc['BMI', :]
plt.figure(figsize=(6, 6))
sns.boxplot(x='Outcome', y='BMI', data=df_copy)

t.loc['DiabetesPedigreeFunction', :]
plt.figure(figsize=(6, 6))
sns.boxplot(x='Outcome', y='DiabetesPedigreeFunction', data=df_copy)

t.loc['Age', :]
plt.figure(figsize=(6, 6))
sns.boxplot(x='Outcome', y='Age', data=df_copy)
plt.ylim([10, 90])

(fig, ax) = plt.subplots(3, 3, figsize=(20, 20))
features = list(df.columns)
(i, j) = (0, 0)
for feature in features:
    if feature == 'Outcome':
        continue
    sns.kdeplot(x=feature, data=df_copy, ax=ax[i][j], hue='Outcome', fill=True)
    i += 1
    if i >= 3:
        i = 0
        j += 1
sns.countplot(x='Outcome', data=df_copy, ax=ax[2][2])

ml = df_copy.copy()
list(ml.columns)
sns.kdeplot(ml['Pregnancies'], fill=True)

sns.kdeplot(np.log(1 + ml['Pregnancies']), fill=True)

ml['Pregnancies'] = np.log(1 + ml['Pregnancies'])
sns.kdeplot(ml['SkinThickness'], fill=True)

sns.kdeplot(np.log(ml['SkinThickness']), fill=True)

ml['SkinThickness'] = np.log(ml['SkinThickness'])
sns.kdeplot(ml['Insulin'], fill=True)

sns.kdeplot(np.log(ml['Insulin']), fill=True)

ml['Insulin'] = np.log(ml['Insulin'])
sns.kdeplot(ml['BMI'], fill=True)

sns.kdeplot(np.log(ml['BMI']), fill=True)

ml['BMI'] = np.log(ml['BMI'])
sns.kdeplot(ml['DiabetesPedigreeFunction'], fill=True)

sns.kdeplot(np.log(ml['DiabetesPedigreeFunction']), fill=True)

ml['DiabetesPedigreeFunction'] = np.log(ml['DiabetesPedigreeFunction'])
sns.kdeplot(ml['Age'], fill=True)

sns.kdeplot(np.log(ml['Age'] - 15), fill=True)

ml['Age'] = np.log(ml['Age'] - 15)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline
X = ml.drop('Outcome', axis=1)
Y = ml['Outcome']
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, random_state=42)
print('X_train: ', np.shape(x_train))
print('y_train: ', np.shape(y_train))
print('X_test: ', np.shape(x_test))
print('y_test: ', np.shape(y_test))
x_train = scale(x_train)
x_test = scale(x_test)
knn = KNeighborsClassifier()