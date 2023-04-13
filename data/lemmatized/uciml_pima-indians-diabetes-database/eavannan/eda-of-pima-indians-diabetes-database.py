import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
file = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
diabetes = pd.read_csv(file)
diabetes.head()
diabetes.shape
diabetes.describe(include='all')
diabetes.isnull().sum()
(diabetes == 0).astype(int).sum(axis=0)
for col in diabetes.iloc[:, 1:6]:
    diabetes[col].replace(0, np.nan, inplace=True)
import missingno as msno
sorted = diabetes.sort_values('Insulin')
msno.matrix(sorted)
for col in diabetes.iloc[:, [1, 5]]:
    diabetes[col].replace(np.nan, diabetes[col].mean(), inplace=True)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='most_frequent')
columns = ['BloodPressure', 'SkinThickness', 'Insulin']
for col in columns:
    diabetes[col] = imp.fit_transform(diabetes[col].values.reshape(-1, 1))
diabetes.info()
Glucose_conditions = [diabetes['Glucose'] <= 50, (diabetes['Glucose'] > 50) & (diabetes['Glucose'] <= 100), (diabetes['Glucose'] > 100) & (diabetes['Glucose'] <= 150), (diabetes['Glucose'] > 150) & (diabetes['Glucose'] <= 200)]
BMI_conditions = [diabetes['BMI'] <= 20, (diabetes['BMI'] > 20) & (diabetes['BMI'] <= 40), (diabetes['BMI'] > 40) & (diabetes['BMI'] <= 60), (diabetes['BMI'] > 60) & (diabetes['BMI'] <= 80)]
Age_conditions = [(diabetes['Age'] >= 20) & (diabetes['Age'] < 30), (diabetes['Age'] >= 30) & (diabetes['Age'] < 40), (diabetes['Age'] >= 40) & (diabetes['Age'] < 50), (diabetes['Age'] >= 50) & (diabetes['Age'] < 60), (diabetes['Age'] >= 60) & (diabetes['Age'] < 70), (diabetes['Age'] >= 70) & (diabetes['Age'] < 80), (diabetes['Age'] >= 80) & (diabetes['Age'] < 90)]
Glucose_values = ['0-50', '51-100', '101-150', '151-200']
BMI_values = ['0-20', '21-40', '41-60', '61-80']
Age_values = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']
diabetes['GlucoseRange'] = np.select(Glucose_conditions, Glucose_values)
diabetes['BMIRange'] = np.select(BMI_conditions, BMI_values)
diabetes['AgeRange'] = np.select(Age_conditions, Age_values)
Glucose_level = CategoricalDtype(categories=Glucose_values, ordered=True)
BMI_level = CategoricalDtype(categories=BMI_values, ordered=True)
Age_level = CategoricalDtype(categories=Age_values, ordered=True)
diabetes['GlucoseRange'] = diabetes['GlucoseRange'].astype(Glucose_level)
diabetes['BMIRange'] = diabetes['BMIRange'].astype(BMI_level)
diabetes['AgeRange'] = diabetes['AgeRange'].astype(Age_level)
print(diabetes.dtypes)
diabetes.hist(figsize=(10, 10))
matrix = diabetes.corr()
mask = np.triu(np.ones_like(matrix, dtype=bool))
pass
pass
pass
cross = pd.crosstab(diabetes.GlucoseRange, diabetes.BMIRange, values=diabetes.Outcome, aggfunc='sum', margins=True, margins_name='Total', normalize='all')
pass
pass
cross = pd.crosstab(diabetes.GlucoseRange, diabetes.AgeRange, values=diabetes.Outcome, aggfunc='sum', margins=True, margins_name='Total', normalize='all')
pass
pass
SkinThickness_jitter = diabetes.SkinThickness + np.random.normal(0, 2, len(diabetes.SkinThickness))
insulin_jitter = diabetes.Insulin + np.random.normal(0, 2, len(diabetes.Insulin))
BloodPressure_jitter = diabetes.BloodPressure + np.random.normal(0, 2, len(diabetes.BloodPressure))
pass
axs[0, 0].plot(diabetes.Age, diabetes.Pregnancies, marker='o', linestyle='', markersize=1.2, alpha=0.8)
axs[0, 0].set_title('Age vs Pregnancies')
axs[0, 1].plot(diabetes.BMI, SkinThickness_jitter, marker='o', linestyle='', markersize=1.1, alpha=0.9)
axs[0, 1].set_title('BMI vs SkinThickness')
axs[1, 0].plot(diabetes.Glucose, insulin_jitter, marker='o', linestyle='', markersize=1.1, alpha=0.9)
axs[1, 0].set_title('Glucose vs Insulin')
axs[1, 1].plot(diabetes.BMI, BloodPressure_jitter, marker='o', linestyle='', markersize=1.2, alpha=0.8)
axs[1, 1].set_title('BMI vs BloodPressure')
axs[2, 0].plot(insulin_jitter, SkinThickness_jitter, marker='o', linestyle='', markersize=1.2, alpha=0.8)
axs[2, 0].set_title('Insulin vs SkinThickness')
fig.tight_layout()
diabetes = diabetes.drop(['GlucoseRange', 'BMIRange', 'AgeRange'], axis=1)
pass
plotnumber = 1
for col in diabetes.iloc[:, 0:8]:
    pass
    pass
    pass
    plotnumber += 1
from scipy import stats
z = np.abs(stats.zscore(diabetes))
threshold = 3
np.where(z > threshold)
diabetes_cleaned = diabetes[(z < 3).all(axis=1)]
print(diabetes_cleaned.shape)
print(diabetes.shape)
pass
X = diabetes_cleaned.drop(['Outcome'], axis=1).values
y = diabetes_cleaned['Outcome'].values
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
knn = KNeighborsClassifier()
steps = [('imputation', imp), ('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, stratify=y, random_state=21)
parameters = {'knn__n_neighbors': np.arange(1, 50)}
knn_cv = GridSearchCV(pipeline, param_grid=parameters, cv=5)