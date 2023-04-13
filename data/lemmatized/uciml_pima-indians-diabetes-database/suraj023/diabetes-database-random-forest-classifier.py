import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.shape
data.isnull().values.any()
import seaborn as sns
corrmat = data.corr()
top_corr_features = corrmat.index
pass
data.corr()
diabetes_true_count = len(data.loc[data['Outcome'] == True])
diabetes_false_count = len(data.loc[data['Outcome'] == False])
(diabetes_true_count, diabetes_false_count)
from sklearn.model_selection import train_test_split
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
predicted_class = ['Outcome']
X = data[feature_columns].values
y = data[predicted_class].values
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=10)
print('total number of rows : {0}'.format(len(data)))
print('number of rows missing Pregnancies: {0}'.format(len(data.loc[data['Pregnancies'] == 0])))
print('number of rows missing glucose_conc: {0}'.format(len(data.loc[data['Glucose'] == 0])))
print('number of rows missing diastolic_bp: {0}'.format(len(data.loc[data['BloodPressure'] == 0])))
print('number of rows missing insulin: {0}'.format(len(data.loc[data['SkinThickness'] == 0])))
print('number of rows missing bmi: {0}'.format(len(data.loc[data['Insulin'] == 0])))
print('number of rows missing diab_pred: {0}'.format(len(data.loc[data['BMI'] == 0])))
print('number of rows missing age: {0}'.format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))
from sklearn.impute import SimpleImputer
fill_values = SimpleImputer(missing_values=0, strategy='mean')
X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)