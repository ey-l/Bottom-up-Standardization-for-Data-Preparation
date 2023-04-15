import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.stats as stats
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings(action='ignore')

def out_remove(col_name, df, cond, m):
    quartile1 = df[col_name].quantile(0.25)
    quartile3 = df[col_name].quantile(0.75)
    iqr = quartile3 - quartile1
    upper = quartile3 + m * iqr
    lower = quartile1 - m * iqr
    if cond == 'both':
        new_df = df[(df[col_name] < upper) & (df[col_name] > lower)]
    elif cond == 'lower':
        new_df = df[df[col_name] > lower]
    else:
        new_df = df[df[col_name] < upper]
    return new_df
diabetes_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_df.head()
diabetes_df.shape
diabetes_df.info()
(fig, (ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
sns.heatmap(diabetes_df.isnull(), cbar=False, ax=ax1)
percent_missing = diabetes_df.isnull().mean() * 100
sns.barplot(x=percent_missing.index, y=percent_missing, ax=ax2)
plt.xticks(rotation=90)

num_cols = diabetes_df.columns
rcParams['figure.figsize'] = (5, 5)
sns.countplot(diabetes_df['Outcome'], palette=['#FC766AFF', '#5B84B1FF']).set_title('Distribution of Outcome')
diabetes_df.describe().T
d_copy = diabetes_df.copy()
d_copy = d_copy.drop(columns=['Outcome'], axis=1)
d_copy = d_copy.replace(0, np.nan)
(fig, (ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
sns.heatmap(d_copy.isnull(), cbar=False, ax=ax1)
percent_missing = d_copy.isnull().mean() * 100
sns.barplot(x=percent_missing.index, y=percent_missing, ax=ax2)
plt.xticks(rotation=90)

X = diabetes_df.iloc[:, :-1]
y = diabetes_df.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
X_train = pd.concat([X_train, y_train], axis=1)
X_test = pd.concat([X_test, y_test], axis=1)


print(X_train.shape)
print(y_train.shape)
rcParams['figure.figsize'] = (30, 15)
sns.set(font_scale=1.5)
sns.set_style('white')
plt.subplots_adjust(hspace=1)
(fig, axes) = plt.subplots(2, 4)
for i in range(4):
    sns.distplot(X_train[num_cols[i]], ax=axes[0, i], rug=True, color='darkblue')
    sns.despine()
for i in range(4, 8):
    sns.distplot(X_train[num_cols[i]], ax=axes[1, i - 4], rug=True, color='darkblue')
    sns.despine()
(fig, axes) = plt.subplots(1, 3)
for i in range(1):
    sns.distplot(X_train[num_cols[0]], ax=plt.subplot(2, 3, 1), rug=True, color='Aqua')
    sns.boxplot(X_train[num_cols[0]], ax=plt.subplot(2, 3, 2), color='Aqua')
    stats.probplot(X_train[num_cols[0]], plot=plt.subplot(2, 3, 3))
print(X_train.shape)
print(y_train.shape)
X_train = out_remove('Pregnancies', X_train, 'both', 1.5)
X_test = out_remove('Pregnancies', X_test, 'both', 1.5)
X_train
(fig, axes) = plt.subplots(1, 3)
for i in range(1):
    sns.distplot(X_train[num_cols[1]], ax=plt.subplot(2, 3, 1), rug=True, color='Violet')
    sns.boxplot(X_train[num_cols[1]], ax=plt.subplot(2, 3, 2), color='Violet')
    stats.probplot(X_train[num_cols[1]], plot=plt.subplot(2, 3, 3))
X_train['Glucose'] = X_train['Glucose'].replace(0, X_train['Glucose'].mean())
X_test['Glucose'] = X_test['Glucose'].replace(0, X_test['Glucose'].mean())
X_train = out_remove('Glucose', X_train, 'both', 1.5)
X_test = out_remove('Glucose', X_test, 'both', 1.5)
(fig, axes) = plt.subplots(1, 3)
for i in range(1):
    sns.distplot(X_train[num_cols[2]], ax=plt.subplot(2, 3, 1), rug=True, color='DarkOrange')
    sns.boxplot(X_train[num_cols[2]], ax=plt.subplot(2, 3, 2), color='DarkOrange')
    stats.probplot(X_train[num_cols[2]], plot=plt.subplot(2, 3, 3))
X_train['BloodPressure'] = X_train['BloodPressure'].replace(0, X_train['BloodPressure'].median())
X_test['BloodPressure'] = X_test['BloodPressure'].replace(0, X_test['BloodPressure'].median())
X_train = out_remove('BloodPressure', X_train, 'lower', 1.5)
X_test = out_remove('BloodPressure', X_test, 'lower', 1.5)
(fig, axes) = plt.subplots(1, 3)
for i in range(1):
    sns.distplot(X_train[num_cols[3]], ax=plt.subplot(2, 3, 1), rug=True, color='blue')
    sns.boxplot(X_train[num_cols[3]], ax=plt.subplot(2, 3, 2), color='blue')
    stats.probplot(X_train[num_cols[3]], plot=plt.subplot(2, 3, 3))
X_train['SkinThickness'] = X_train['SkinThickness'].replace(0, X_train['SkinThickness'].mean())
X_test['SkinThickness'] = X_test['SkinThickness'].replace(0, X_test['SkinThickness'].mean())
X_train = out_remove('SkinThickness', X_train, 'both', 1.5)
X_test = out_remove('SkinThickness', X_test, 'both', 1.5)
(fig, axes) = plt.subplots(1, 3)
for i in range(1):
    sns.distplot(X_train[num_cols[4]], ax=plt.subplot(2, 3, 1), rug=True, color='black')
    sns.boxplot(X_train[num_cols[4]], ax=plt.subplot(2, 3, 2), color='black')
    stats.probplot(X_train[num_cols[4]], plot=plt.subplot(2, 3, 3))
X_train['Insulin'] = X_train['Insulin'].replace(0, X_train['Insulin'].median())
X_test['Insulin'] = X_test['Insulin'].replace(0, X_test['Insulin'].median())
X_train = out_remove('Insulin', X_train, 'both', 1.5)
X_test = out_remove('Insulin', X_test, 'both', 1.5)
(fig, axes) = plt.subplots(1, 3)
for i in range(1):
    sns.distplot(X_train[num_cols[5]], ax=plt.subplot(2, 3, 1), rug=True, color='green')
    sns.boxplot(X_train[num_cols[5]], ax=plt.subplot(2, 3, 2), color='green')
    stats.probplot(X_train[num_cols[5]], plot=plt.subplot(2, 3, 3))
X_train['BMI'] = X_train['BMI'].replace(0, X_train['BMI'].mean())
X_test['BMI'] = X_test['BMI'].replace(0, X_test['BMI'].mean())
(fig, axes) = plt.subplots(1, 3)
for i in range(1):
    sns.distplot(X_train[num_cols[6]], ax=plt.subplot(2, 3, 1), rug=True, color='brown')
    sns.boxplot(X_train[num_cols[6]], ax=plt.subplot(2, 3, 2), color='brown')
    stats.probplot(X_train[num_cols[6]], plot=plt.subplot(2, 3, 3))
X_train = out_remove('DiabetesPedigreeFunction', X_train, 'both', 1.5)
X_test = out_remove('DiabetesPedigreeFunction', X_test, 'both', 1.5)
(fig, axes) = plt.subplots(1, 3)
for i in range(1):
    sns.distplot(X_train[num_cols[7]], ax=plt.subplot(2, 3, 1), rug=True, color='y')
    sns.boxplot(X_train[num_cols[7]], ax=plt.subplot(2, 3, 2), color='y')
    stats.probplot(X_train[num_cols[7]], plot=plt.subplot(2, 3, 3))
rcParams['figure.figsize'] = (30, 15)
sns.set(font_scale=1.5)
sns.set_style('white')
plt.subplots_adjust(hspace=1)
(fig, axes) = plt.subplots(2, 4)
for i in range(4):
    sns.distplot(X_train[num_cols[i]], ax=axes[0, i], rug=True, color='brown')
    sns.despine()
for i in range(4, 8):
    sns.distplot(X_train[num_cols[i]], ax=axes[1, i - 4], rug=True, color='brown')
    sns.despine()
sns.heatmap(X_train.corr(), cmap='RdGy', annot=True, cbar=True)
plt.figure(figsize=(8, 4), dpi=110)
sns.scatterplot(data=X_train, x='BMI', y='SkinThickness', color='darkblue')
plt.plot([18, 47], [0, 55], 'red', linewidth=4)
plt.xlim(20, 60)
plt.ylim(4, 60)
plt.figure(figsize=(8, 4), dpi=110)
sns.scatterplot(data=X_train, x='Age', y='Pregnancies', color='darkblue')
plt.plot([0, 80], [1, 12], 'red', linewidth=4)
plt.ylim(0, 18)
plt.xlim(18, 90)

from sklearn.tree import DecisionTreeClassifier
plt.figure(figsize=(10, 5))
model = DecisionTreeClassifier()
z_copy = X_train.copy()
y = z_copy.iloc[:, -1]
X = z_copy.iloc[:, :-1]