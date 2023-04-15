import numpy as np
import pandas as pd
import sklearn
path = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
diabetes_data = pd.read_csv(path)
print('Reading completed')
diabetes_data.describe()
diabetes_data.shape
y = diabetes_data.Outcome
y.head()
diabetes_data.isna().any()
from seaborn import heatmap
matrix_corr = diabetes_data.corr()
heatmap(matrix_corr, annot=True)
features = ['Glucose', 'BMI', 'Insulin', 'Age', 'Pregnancies', 'DiabetesPedigreeFunction', 'SkinThickness', 'BloodPressure']
X = diabetes_data[features]
X.head()
from sklearn.model_selection import train_test_split
(train_X, val_X, train_y, val_y) = train_test_split(X, y, random_state=0)
print('Splited')
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear', random_state=0)