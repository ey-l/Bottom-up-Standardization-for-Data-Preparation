import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
diabetes_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_df
diabetes_df.info()
sns.countplot(diabetes_df['Outcome'])
diabetes_df.describe()
cols = diabetes_df.columns
print(cols)
(fig, axes) = plt.subplots(3, 3, figsize=(10, 10), gridspec_kw=dict(hspace=0.5, wspace=0.6))
fig.suptitle('Frequency plot for different column values')
for (col, az) in zip(cols, axes.flat):
    sns.histplot(diabetes_df[col], ax=az)
(diabetes_df[cols] == 0).sum()
diabetes_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
(diabetes_df[cols] == 0).sum()
diabetes_df['BloodPressure'] = diabetes_df['BloodPressure'].fillna(diabetes_df['BloodPressure'].mean())
diabetes_df['BMI'] = diabetes_df['BMI'].fillna(diabetes_df['BMI'].mean())
diabetes_df['Glucose'] = diabetes_df['Glucose'].fillna(diabetes_df['Glucose'].median())
diabetes_df['SkinThickness'] = diabetes_df['SkinThickness'].fillna(diabetes_df['SkinThickness'].median())
diabetes_df['Insulin'] = diabetes_df['Insulin'].fillna(diabetes_df['Insulin'].median())
diabetes_df[diabetes_df['SkinThickness'] > 90]
diabetes_df['SkinThickness'].loc[579] = diabetes_df['SkinThickness'].median()
diabetes_df.loc[579]
diabetes_df[(diabetes_df['Pregnancies'] > 10) & (diabetes_df['Age'] < 30)].shape
diabetes_df.describe()
sns.pairplot(diabetes_df, hue='Outcome', height=2)
plt.figure(figsize=(10, 10))
sns.heatmap(diabetes_df.corr(), square=True, linewidths=0.5, annot=True, cbar=False)
from sklearn.model_selection import train_test_split
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, stratify=y)
print('X_train shape : ', X_train.shape)
print('y_train shape : ', y_train.shape)
print('X_test shape  : ', X_test.shape)
print('y_test shape  : ', y_test.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()