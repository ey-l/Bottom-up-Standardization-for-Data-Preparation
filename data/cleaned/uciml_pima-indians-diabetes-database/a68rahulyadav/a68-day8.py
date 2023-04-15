import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
ds = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
no_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
for column in no_zero:
    ds[column] = ds[column].replace(0, np.NaN)
    mean = int(ds[column].mean(skipna=True))
    ds[column] = ds[column].replace(np.NaN, mean)
X = ds.iloc[:, 0:8]
y = ds.iloc[:, 8]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0, test_size=0.2)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(2, 15)}
knn_gscv = GridSearchCV(knn, param_grid, cv=5)