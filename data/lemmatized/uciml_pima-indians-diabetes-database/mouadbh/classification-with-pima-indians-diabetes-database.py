import numpy as np
import pandas as pd
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
col = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
label = df['Outcome'].values
features = df[list(col)].values
label
features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(features, label, test_size=0.3)
clf = RandomForestClassifier(n_estimators=10)