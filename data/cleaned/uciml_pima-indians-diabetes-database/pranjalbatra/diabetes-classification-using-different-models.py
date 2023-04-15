import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

sns.set()
import warnings
warnings.filterwarnings('ignore')
os.getcwd()
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.sample(6)
df.info()
df.isnull().sum()
df.describe().T
df.hist(grid=False, figsize=(12, 16))
plt.tight_layout()

df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].median())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())
df.hist(grid=False, figsize=(12, 16))
plt.tight_layout()

(fig, ax) = plt.subplots()
for cols in df.columns:
    sns.boxplot(y=cols, data=df)
    plt.ylabel(cols)

for col in df.columns:
    sns.kdeplot(df, x=df[col])
    plt.tight_layout()

df.corr()['Outcome'].sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='viridis_r')
sns.pairplot(df, hue='Outcome')
plt.figure(figsize=(5, 5))
plt.pie(df['Outcome'].value_counts(), labels=['Non-diabetic', 'Diabetic'], radius=1, autopct='%1.1f%%', explode=[0, 0.1], labeldistance=1.15, startangle=90)
plt.legend(title='Outcome:', loc='upper right', bbox_to_anchor=(1.6, 1))

print(df['Outcome'].value_counts())
print('\n\n')
print(sns.countplot(x=df['Outcome'], saturation=0.8, hue=df['Outcome']))
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.25, random_state=101, stratify=y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
pd.DataFrame(x_train_scaled).describe().T
for col in pd.DataFrame(x_train_scaled):
    sns.kdeplot(pd.DataFrame(x_train_scaled), x=pd.DataFrame(x_train_scaled)[col])
    plt.tight_layout()

from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(random_state=42)