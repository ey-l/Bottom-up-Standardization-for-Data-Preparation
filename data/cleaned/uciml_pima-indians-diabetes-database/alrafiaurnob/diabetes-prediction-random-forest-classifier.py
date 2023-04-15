import numpy as np
import pandas as pd
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data
data.isnull().sum()
import seaborn as sns
import matplotlib.pyplot as plt
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap='RdYlGn')
data.corr()
diabetes_true_count = len(data.loc[data['Outcome'] == True])
diabetes_false_count = len(data.loc[data['Outcome'] == False])
(diabetes_true_count, diabetes_false_count)
from sklearn.model_selection import train_test_split
x = data.iloc[:, :-1]
x
y = data.iloc[:, -1:]
y
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=10)
print('total number of rows : {0}'.format(len(data)))
print('number of rows missing Pregnancies: {0}'.format(len(data.loc[data['Pregnancies'] == 0])))
print('number of rows missing Glucose: {0}'.format(len(data.loc[data['Glucose'] == 0])))
print('number of rows missing BloodPressure: {0}'.format(len(data.loc[data['BloodPressure'] == 0])))
print('number of rows missing SkinThickness: {0}'.format(len(data.loc[data['SkinThickness'] == 0])))
print('number of rows missing Insulin: {0}'.format(len(data.loc[data['Insulin'] == 0])))
print('number of rows missing DiabetesPedigreeFunction: {0}'.format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))
print('number of rows missing age: {0}'.format(len(data.loc[data['Age'] == 0])))
from sklearn.impute import SimpleImputer
fill_values = SimpleImputer(missing_values=0, strategy='mean')
x_train = fill_values.fit_transform(x_train)
x_test = fill_values.fit_transform(x_test)
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)