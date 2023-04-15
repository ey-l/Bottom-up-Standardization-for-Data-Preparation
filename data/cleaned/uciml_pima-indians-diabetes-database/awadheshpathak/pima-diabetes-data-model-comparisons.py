import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import copy
data_raw = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data_raw.dtypes
data_raw.shape
data_raw.sample(5)
data_raw.info()
data_raw.describe()
data_raw.boxplot(figsize=(10, 10), rot=90)
data_raw.hist(figsize=(15, 20))
not_allowed_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data = copy.deepcopy(data_raw)
data[not_allowed_zero_cols] = data[not_allowed_zero_cols].replace(0, np.NaN)
data.isnull().sum()
(fig, ax) = plt.subplots(nrows=3, ncols=2, figsize=(15, 20))
sns.distplot(data.Glucose, ax=ax[0][0])
sns.distplot(data.BloodPressure, ax=ax[0][1])
sns.distplot(data.Insulin, ax=ax[1][0])
sns.distplot(data.SkinThickness, ax=ax[1][1])
sns.distplot(data.BMI, ax=ax[2][0])
data['Glucose'].fillna(data.Glucose.mean(), inplace=True)
data['BloodPressure'].fillna(data.BloodPressure.mean(), inplace=True)
data['BMI'].fillna(data.BMI.mean(), inplace=True)
data['SkinThickness'].fillna(data.SkinThickness.mean(), inplace=True)
data['Insulin'].fillna(data.Insulin.median(), inplace=True)
(fig, ax) = plt.subplots(nrows=3, ncols=2, figsize=(15, 20))
sns.distplot(data.Glucose, ax=ax[0][0])
sns.distplot(data.BloodPressure, ax=ax[0][1])
sns.distplot(data.Insulin, ax=ax[1][0])
sns.distplot(data.SkinThickness, ax=ax[1][1])
sns.distplot(data.BMI, ax=ax[2][0])
data.dtypes.value_counts().plot(kind='bar')
sns.countplot(data.Outcome).set(title='Data Imbalance Check')
sns.pairplot(data, hue='Outcome')
cor = data.corr()
mask = np.triu(np.ones_like(cor, dtype=np.bool))
plt.figure(figsize=(10, 10))
sns.heatmap(cor, mask=mask, center=0, square=True, linewidths=0.5, annot=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, mean_squared_error, roc_auc_score, confusion_matrix, roc_curve, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(data.drop(['Outcome'], axis='columns'))
(X_train, X_test, y_train, y_test) = train_test_split(X_scaled, data.Outcome, random_state=123, test_size=0.2)
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(class_weight='balanced', random_state=123, max_iter=500)