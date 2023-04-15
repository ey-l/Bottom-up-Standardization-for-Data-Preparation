import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn import metrics
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df.shape
df.head()
df.isnull().values.any()
columns = list(df)[0:-1]
df[columns].hist(bins=80, figsize=(12, 50), layout=(14, 4))
df.rename(columns={'DiabetesPedigreeFunction': 'DPF'}, inplace=True)
df.rename(columns={'BloodPressure': 'BP', 'SkinThickness': 'SkinTh'}, inplace=True)
df.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), vmax=1, square=True, annot=True, cmap='Blues')
plt.title('Correlation between different attributes')

sns.pairplot(df, diag_kind='kde')
df['Outcome'].value_counts()
df.head()
X = df.drop('Outcome', axis=1)
y = df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=1)
print('{}% data is in the training set'.format(len(X_train) / len(df) * 100))
print('{}% data is in the testing set'.format(len(X_test) / len(df) * 100))
X_train.head()
X_train.describe().T
replace_ = SimpleImputer(missing_values=0, strategy='mean')
cols = X_train.columns
X_train = pd.DataFrame(replace_.fit_transform(X_train))
X_test = pd.DataFrame(replace_.fit_transform(X_test))
X_train.columns = cols
X_test.columns = cols
X_train.head()
model = LogisticRegression(solver='liblinear')