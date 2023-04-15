import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe().T
df_copy = df.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(df_copy.isnull().sum())
p = df.hist(figsize=(12, 12))
df_copy['Glucose'].fillna(df_copy['Glucose'].median(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].median(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)
p = df_copy.hist(figsize=(12, 12))
print(df_copy['Outcome'].value_counts())
p = df_copy['Outcome'].value_counts().plot(kind='bar')
p = sns.pairplot(df_copy, hue='Outcome')
plt.figure(figsize=(10, 10))
p = sns.heatmap(df_copy.corr(), annot=True, cmap='RdYlGn')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(df_copy.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
y = df_copy['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=1 / 3, random_state=42, stratify=y)
from sklearn.neighbors import KNeighborsClassifier
train_scores = []
test_scores = []
for i in range(1, 15):
    knn = KNeighborsClassifier(i)