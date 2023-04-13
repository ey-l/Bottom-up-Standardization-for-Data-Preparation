import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.columns
data.shape
data.describe().T
data.info()
diabetes_data_copy = data.copy(deep=True)
diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(diabetes_data_copy.isnull().sum())
p = data.hist(figsize=(20, 20))
pass
corr = data.corr()
corr.style.background_gradient()
pass
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
scaled_data_X = pd.DataFrame(sc_X.fit_transform(data.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
scaled_data_Y = data.Outcome
scaled_data_X.head()
scaled_data_X.hist(figsize=(20, 10))
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(scaled_data_X, scaled_data_Y, train_size=0.8, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i)