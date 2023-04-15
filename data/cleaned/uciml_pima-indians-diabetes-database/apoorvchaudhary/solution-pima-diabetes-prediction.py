import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.shape
data.head()
data.isnull().values.any()
import seaborn as sns
import matplotlib.pyplot as plt
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10, 12))
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap='RdYlGn')
data.corr()
diabetes_true_count = len(data.loc[data['Outcome'] == 1])
diabetes_false_count = len(data.loc[data['Outcome'] == 0])
from sklearn.model_selection import train_test_split
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
predicted_class = ['Outcome']
X = data[feature_columns].values
y = data[predicted_class].values
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=10)
print('Total number of rows : {0}'.format(len(data)))
print('Number of rows missing glucose Pregnancies : {0}'.format(len(data.loc[data['Pregnancies'] == 0])))
print('Number of rows missing glucose Glucose : {0}'.format(len(data.loc[data['Glucose'] == 0])))
print('Number of rows missing glucose BloodPressure : {0}'.format(len(data.loc[data['BloodPressure'] == 0])))
print('Number of rows missing glucose SkinThickness : {0}'.format(len(data.loc[data['SkinThickness'] == 0])))
print('Number of rows missing glucose Insulin : {0}'.format(len(data.loc[data['Insulin'] == 0])))
print('Number of rows missing glucose BMI : {0}'.format(len(data.loc[data['BMI'] == 0])))
print('Number of rows missing glucose DiabetesPedigreeFunction : {0}'.format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))
print('Number of rows missing glucose Age : {0}'.format(len(data.loc[data['Age'] == 0])))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('std_scaler', StandardScaler())])
X_transform = my_pipeline.fit_transform(X_train)
X_transform.shape
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=0)