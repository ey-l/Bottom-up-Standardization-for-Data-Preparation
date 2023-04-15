import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.shape
data.info()
print(data.describe())
cols_to_impute = ['Glucose', 'SkinThickness', 'Insulin', 'BMI', 'BloodPressure']
data_copy = data.copy(deep=True)
data[cols_to_impute] = data[cols_to_impute].replace(0, np.NaN)
data.isnull().sum()
data.hist(figsize=(20, 20))
data.boxplot(figsize=(15, 10))
data.fillna(data.median(), inplace=True)
data.isnull().sum()
data.hist(figsize=(20, 20))
sns.pairplot(data, hue='Outcome')
corrmat = data.corr()
(f, ax) = plt.subplots(figsize=(9, 8))
sns.heatmap(corrmat, ax=ax, cmap='YlGnBu', linewidths=0.1, annot=True)
top_corr_features = corrmat.index
ax = sns.countplot(data['Outcome'])
for p in ax.patches:
    ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 0.01))
from sklearn.model_selection import train_test_split
feature_col = list(data.iloc[:, :-1].columns)
outcome_col = ['Outcome']
print('Feature columns = ', feature_col)
print('Outcome columns = ', outcome_col)
X = data[feature_col].values
y = data[outcome_col].values
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=10)
from sklearn.linear_model import LogisticRegression