import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
diab = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diab.head()
diab.shape
diab.info()
diab.duplicated().sum()
diab.describe().T
diab.corr().style.background_gradient(cmap='coolwarm')
diab.columns
pass
pass
pass
pass
diab['Outcome'].value_counts()
x = diab.drop('Outcome', axis=1)
y = diab['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(x, y)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=17, metric='manhattan')