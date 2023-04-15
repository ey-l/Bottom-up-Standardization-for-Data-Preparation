import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(diabetes.columns)
diabetes.head()
print('dimension of diabetes data: {}'.format(diabetes.shape))
print(diabetes.groupby('Outcome').size())
import seaborn as sns
sns.countplot(diabetes['Outcome'], label='Count')
diabetes.info()
diabetes.describe().transpose()
colormap = plt.cm.viridis
plt.figure(figsize=(15, 15))
plt.title('Pearson Correlation of attributes', y=1.05, size=19)
sns.heatmap(diabetes.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
spd = pd.plotting.scatter_matrix(diabetes, figsize=(20, 20), diagonal='kde')
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=11)
X_train.shape
from sklearn.svm import SVC
svc = SVC()