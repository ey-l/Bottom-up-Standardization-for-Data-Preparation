import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
diabetes_dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_dataset.head()
diabetes_dataset.shape
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()
diabetes_dataset.isnull().sum()
diabetes_dataset.BloodPressure.hist(bins=20)
diabetes_dataset.boxplot(column='BloodPressure')
upper_limit = np.percentile(diabetes_dataset['BloodPressure'], 95)
print(upper_limit)
diabetes_dataset['Outlier_flag'] = np.where(diabetes_dataset['BloodPressure'] > upper_limit, 'Outlier', 'Not_Outlier')
diabetes_dataset['Outlier_flag'].value_counts()
diabetes_dataset['BloodPressure'] = np.where(diabetes_dataset['BloodPressure'] > upper_limit, upper_limit, diabetes_dataset['BloodPressure'])
diabetes_dataset.boxplot(column='BloodPressure')
diabetes_dataset.drop('Outlier_flag', axis=1, inplace=True)
lower_limit = np.percentile(diabetes_dataset['BloodPressure'], 5)
print(lower_limit)
diabetes_dataset['Outlier_flag'] = np.where(diabetes_dataset['BloodPressure'] < lower_limit, 'Outlier', 'Not_Outlier')
diabetes_dataset['Outlier_flag'].value_counts()
diabetes_dataset['BloodPressure'] = np.where(diabetes_dataset['BloodPressure'] < lower_limit, lower_limit, diabetes_dataset['BloodPressure'])
diabetes_dataset.boxplot(column='BloodPressure')
diabetes_dataset.drop('Outlier_flag', axis=1, inplace=True)
X = diabetes_dataset.drop(columns='Outcome')
y = diabetes_dataset['Outcome']
print(X)
print(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
classifier = svm.SVC(kernel='linear')