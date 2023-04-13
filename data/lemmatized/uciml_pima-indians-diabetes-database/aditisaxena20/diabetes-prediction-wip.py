import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.shape
data.info()
data.isnull().values.any()
data.duplicated().any()
import seaborn as sns
correl = data.corr()
corr_features = correl.index
pass
pass
diabetic_check = data['Outcome'].value_counts().reset_index()
diabetic_check
diabetic = diabetic_check['Outcome'][1]
diabetic
non_diabetic = diabetic_check['Outcome'][0]
non_diabetic
(diabetic, non_diabetic)
from sklearn.model_selection import train_test_split
data.columns
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
predicted_class = ['Outcome']
X = data[feature_columns]
Y = data[predicted_class]
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.3, random_state=10)
len(X_train)
data.columns
print('total number of rows : {0}'.format(len(data)))
print("total number of rows missing in 'Pregnancies': {0}".format(sum(data['Pregnancies'] == 0)))
print("total number of rows missing in 'Glucose': {0}".format(sum(data['Glucose'] == 0)))
print("total number of rows missing in 'BloodPressure': {0}".format(sum(data['BloodPressure'] == 0)))
print("total number of rows missing in 'SkinThickness': {0}".format(sum(data['SkinThickness'] == 0)))
print("total number of rows missing in 'Insulin: {0}".format(sum(data['Insulin'] == 0)))
print("total number of rows missing in 'BMI': {0}".format(sum(data['BMI'] == 0)))
print("total number of rows missing in 'DiabetesPedigreeFunction': {0}".format(sum(data['DiabetesPedigreeFunction'] == 0)))
print("total number of rows missing in 'Age': {0}".format(sum(data['Age'] == 0)))
from sklearn.impute import SimpleImputer
fill_values = SimpleImputer(missing_values=0, strategy='mean')
X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)