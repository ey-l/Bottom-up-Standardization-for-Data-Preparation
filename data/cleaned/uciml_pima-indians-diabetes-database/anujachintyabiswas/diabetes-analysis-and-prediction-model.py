import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore
from sklearn.metrics import classification_report
from sklearn import metrics
sns.set(rc={'figure.dpi': 300, 'savefig.dpi': 300})
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
pd.set_option('display.max_columns', 29)
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.tail()
print(f"Columns name in the dataset: {colored(df.columns, 'magenta')}")
print(f"Shape of the dataset: {colored(df.shape, 'magenta')}")
df.isnull().sum()
df.info()
for i in df.columns:
    print(i, len(df[df[i] == 0]))
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
df.isnull().sum()
df = df.fillna(method='ffill')
df = df.fillna(method='bfill')
df.isnull().sum()
(fig1, axes1) = plt.subplots(4, 2, figsize=(14, 25))
list1_col = df.columns
for i in range(len(list1_col) - 1):
    row = i // 2
    col = i % 2
    ax = axes1[row, col]
    sns.boxplot(df[list1_col[i]], ax=ax).set(title=list1_col[i].upper())
print(df.Insulin.shape)
print(df.SkinThickness.shape)
print(df.BMI.shape)
print(df.DiabetesPedigreeFunction.shape)
print(df.Pregnancies.shape)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
df = df[~((df.iloc[:, 4:5] < Q1 - 1.5 * IQR) | (df.iloc[:, 4:5] > Q3 + 1.5 * IQR)).any(axis=1)]
df = df[~((df.iloc[:, 3:4] < Q1 - 1.5 * IQR) | (df.iloc[:, 3:4] > Q3 + 1.5 * IQR)).any(axis=1)]
df = df[~((df.iloc[:, 5:6] < Q1 - 1.5 * IQR) | (df.iloc[:, 5:6] > Q3 + 1.5 * IQR)).any(axis=1)]
df = df[~((df.iloc[:, 6:7] < Q1 - 1.5 * IQR) | (df.iloc[:, 6:7] > Q3 + 1.5 * IQR)).any(axis=1)]
df = df[~((df.iloc[:, 0:1] < Q1 - 1.5 * IQR) | (df.iloc[:, 0:1] > Q3 + 1.5 * IQR)).any(axis=1)]
print(df.Insulin.shape)
print(df.SkinThickness.shape)
print(df.BMI.shape)
print(df.DiabetesPedigreeFunction.shape)
print(df.Pregnancies.shape)
df.shape
(fig1, axes1) = plt.subplots(4, 2, figsize=(14, 25))
list1_col = df.columns
for i in range(len(list1_col) - 1):
    row = i // 2
    col = i % 2
    ax = axes1[row, col]
    sns.boxplot(df[list1_col[i]], ax=ax).set(title=list1_col[i].upper())
df.describe()
df.head()
explode = (0.08, 0)
df['Outcome'].value_counts().plot.pie(autopct='%1.2f%%', figsize=(3, 3), explode=explode, colors=['#99ff99', '#ff6666'])
plt.title('Pie plot of diabetes results', fontsize=14)
plt.tight_layout()
plt.legend()

sns.countplot(x='Outcome', data=df)
df.Outcome.value_counts()
sns.pairplot(df, hue='Outcome', diag_kind='kde')
sns.countplot(x='Pregnancies', data=df)
sns.set(font_scale=0.4)
sns.countplot(x='Age', data=df)
sns.set(font_scale=0.6)
sns.heatmap(df.corr(), annot=True, fmt='.0%')
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])
df.head()
X = df.drop('Outcome', axis=1)
y = df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.head()
y_train.head()
X_test.head()
y_test.head()
model = LogisticRegression()