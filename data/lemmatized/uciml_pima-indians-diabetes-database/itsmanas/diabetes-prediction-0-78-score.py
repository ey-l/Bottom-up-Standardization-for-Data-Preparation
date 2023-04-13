import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.info()
data.describe()
corr_matrix = data.corr()
corr_matrix
import seaborn as sns
pass
import matplotlib.pyplot as plt
corrmat = data.corr()
top_corr_features = corrmat.index
pass
pass
from sklearn.model_selection import train_test_split
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
print('total number of rows : {0}'.format(len(data)))
print('number of rows missing Glucose: {0}'.format(len(data.loc[data['Glucose'] == 0])))
print('number of rows missing BloodPressure: {0}'.format(len(data.loc[data['BloodPressure'] == 0])))
print('number of rows missing Insulin: {0}'.format(len(data.loc[data['Insulin'] == 0])))
print('number of rows missing BMI: {0}'.format(len(data.loc[data['BMI'] == 0])))
print('number of rows missing DiabetesPedigreeFunction: {0}'.format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))
print('number of rows missing Age: {0}'.format(len(data.loc[data['Age'] == 0])))
print('number of rows missing SkinThickness: {0}'.format(len(data.loc[data['SkinThickness'] == 0])))
from sklearn.impute import SimpleImputer
fill_values = SimpleImputer(missing_values=0, strategy='mean')
X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=10)