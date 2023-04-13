import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, roc_curve, roc_auc_score
Data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
pd.set_option('display.max_columns', 9)
Data.head()
Data.shape
Missing_values_percent = 100 * (Data.isnull().sum() / len(Data['Insulin']))
print(Missing_values_percent)
Data.dtypes
sbn.pairplot(Data, hue='Outcome')
sbn.pairplot(Data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']])
Data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']].hist(figsize=(16, 9), color='darkred', bins=60, grid=False, layout=(2, 3))
sbn.displot(data=Data, x='BMI', kind='kde')
sbn.displot(data=Data, x='Glucose', kind='kde')
pass
sbn.heatmap(Data.corr(), annot=True, vmin=-1, cmap='coolwarm')
pass
triu = np.triu(Data.corr())
sbn.heatmap(Data.corr(), annot=True, vmin=-1, cmap='coolwarm', mask=triu)
pass
tril = np.tril(Data.corr())
sbn.heatmap(Data.corr(), annot=True, vmin=-1, cmap='coolwarm', mask=tril)
Norm = MinMaxScaler(feature_range=(0, 1))
x = Data.drop(['Outcome'], axis=1)
x = Norm.fit_transform(x)
y = Data['Outcome']
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.33, random_state=45)
ModelKNN = KNeighborsClassifier()
K_Values = np.array([11, 12, 13, 14, 15, 16, 17, 18])
metric = ['minkowski', 'chebyshev']
p = np.array([5, 6, 7, 8, 9, 10, 11])
param_grid = {'n_neighbors': K_Values, 'metric': metric, 'p': p}
GridKNN = GridSearchCV(estimator=ModelKNN, param_grid=param_grid, cv=5)