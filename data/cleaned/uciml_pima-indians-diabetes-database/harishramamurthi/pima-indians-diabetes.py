import numpy as np
import pandas as pd
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.corr()
df.drop(['SkinThickness'], axis=1, inplace=True)
df
df.skew()
df.DiabetesPedigreeFunction = np.log(df['DiabetesPedigreeFunction'])
df.Age = np.log(df['Age'])
df.BloodPressure = np.sqrt(df['BloodPressure'])
df.Insulin = np.sqrt(df['Insulin'])
df.skew()
y = df.Outcome
col = ['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'BloodPressure']
x = df[col]
from sklearn.model_selection import train_test_split
(X_train, x_test, y_train, y_test) = train_test_split(x, y, train_size=0.8)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()