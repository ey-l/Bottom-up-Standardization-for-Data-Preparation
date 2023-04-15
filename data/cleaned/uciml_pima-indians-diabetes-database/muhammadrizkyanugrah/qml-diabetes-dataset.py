
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
diabetes_dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_dataset.head()
diabetes_dataset.tail()
diabetes_dataset.sample(10)
diabetes_dataset.shape
diabetes_dataset.dtypes
diabetes_dataset.info()
diabetes_dataset.describe()
diabetes_dataset.isnull().sum()
diabetes_dataset.columns
print('No. of zero values in Glucose ', diabetes_dataset[diabetes_dataset['Glucose'] == 0].shape[0])
print('No. of zero values in BloodPressure ', diabetes_dataset[diabetes_dataset['BloodPressure'] == 0].shape[0])
print('No. of zero values in SkinThickness ', diabetes_dataset[diabetes_dataset['SkinThickness'] == 0].shape[0])
print('No. of zero values in Insulin ', diabetes_dataset[diabetes_dataset['Insulin'] == 0].shape[0])
print('No. of zero values in BMI ', diabetes_dataset[diabetes_dataset['BMI'] == 0].shape[0])
diabetes_dataset['Glucose'] = diabetes_dataset['Glucose'].replace(0, diabetes_dataset['Glucose'].mean())
diabetes_dataset['BloodPressure'] = diabetes_dataset['BloodPressure'].replace(0, diabetes_dataset['BloodPressure'].mean())
diabetes_dataset['SkinThickness'] = diabetes_dataset['SkinThickness'].replace(0, diabetes_dataset['SkinThickness'].mean())
diabetes_dataset['Insulin'] = diabetes_dataset['Insulin'].replace(0, diabetes_dataset['Insulin'].mean())
diabetes_dataset['BMI'] = diabetes_dataset['BMI'].replace(0, diabetes_dataset['BMI'].mean())
diabetes_dataset.describe()
(f, ax) = plt.subplots(1, 2, figsize=(10, 5))
diabetes_dataset['Outcome'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Outcome')
ax[0].set_ylabel('')
sns.countplot(x='Outcome', data=diabetes_dataset, ax=ax[1])
ax[1].set_title('Outcome')
(N, P) = diabetes_dataset['Outcome'].value_counts()
print('Negative (0): ', N)
print('Positive (1): ', P)
plt.grid()

diabetes_dataset.hist(bins=10, figsize=(10, 10))

from pandas.plotting import scatter_matrix
scatter_matrix(diabetes_dataset, figsize=(20, 20))
sns.pairplot(data=diabetes_dataset, hue='Outcome')

import seaborn as sns
corrmat = diabetes_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10, 10))
g = sns.heatmap(diabetes_dataset[top_corr_features].corr(), annot=True, cmap='RdYlGn')
target_name = 'Outcome'
labels = diabetes_dataset[target_name]
features = diabetes_dataset.drop(target_name, axis=1)
print('Features =\n', features)
print('labels =\n', labels)
features.head()
labels.head()
from sklearn.preprocessing import MinMaxScaler
features = MinMaxScaler().fit_transform(features)
print('Features =\n', features)
print('labels =\n', labels)
from sklearn.model_selection import train_test_split
from qiskit.utils import algorithm_globals
(train_features, test_features, train_labels, test_labels) = train_test_split(features, labels, train_size=0.8, random_state=123)
from sklearn.svm import SVC
svc = SVC()