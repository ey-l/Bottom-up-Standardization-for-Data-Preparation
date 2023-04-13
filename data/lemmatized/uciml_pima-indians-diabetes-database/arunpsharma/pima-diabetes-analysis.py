import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dbdt = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dbdt.head()
dbdt.info()
dbdt.describe()
dbdt_cp = dbdt.copy(deep=True)
dbdt_cp[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = dbdt_cp[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
dbdt_cp['Glucose'].fillna(dbdt_cp['Glucose'].mean(), inplace=True)
dbdt_cp['BloodPressure'].fillna(dbdt_cp['BloodPressure'].mean(), inplace=True)
dbdt_cp['SkinThickness'].fillna(dbdt_cp['SkinThickness'].mean(), inplace=True)
dbdt_cp['Insulin'].fillna(dbdt_cp['Insulin'].mean(), inplace=True)
dbdt_cp['BMI'].fillna(dbdt_cp['BMI'].mean(), inplace=True)
dbdt_cp.describe()
pass
pass
pass
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(dbdt_cp.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = dbdt_cp['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=1 / 3, random_state=42, stratify=y)
X_train.head()
X_test.head()
y_train.head()
y_test.head()
from sklearn.neighbors import KNeighborsClassifier
test_scores = []
train_scores = []
for i in range(1, 20):
    knn = KNeighborsClassifier(i)