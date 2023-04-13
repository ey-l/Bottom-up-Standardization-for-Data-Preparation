import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
pass
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head(10)
data.shape
data.info()
data.describe()
data.isnull().sum()
import missingno as msno
msno.bar(data)
pass
col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for i in col:
    data[i].replace(0, data[i].mean(), inplace=True)
p = data.hist(figsize=(20, 20))
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
data.var()
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(data.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = data.Outcome
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(X, y, test_size=0.3, random_state=3)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C=1, penalty='l2')