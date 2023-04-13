import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pass
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.describe()
pass
plotnumber = 1
for column in data:
    if plotnumber <= 9:
        pass
        pass
        pass
    plotnumber += 1
data['BMI'] = data['BMI'].replace(0, data['BMI'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0, data['BloodPressure'].mean())
data['Glucose'] = data['Glucose'].replace(0, data['Glucose'].mean())
data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].mean())
pass
plotnumber = 1
for column in data:
    if plotnumber <= 9:
        pass
        pass
        pass
    plotnumber += 1
pass
pass
outlier = data['Pregnancies'].quantile(0.98)
data = data[data['Pregnancies'] < outlier]
outlier = data['BMI'].quantile(0.99)
data = data[data['BMI'] < outlier]
outlier = data['SkinThickness'].quantile(0.99)
data = data[data['SkinThickness'] < outlier]
outlier = data['Insulin'].quantile(0.95)
data = data[data['Insulin'] < outlier]
outlier = data['DiabetesPedigreeFunction'].quantile(0.99)
data = data[data['DiabetesPedigreeFunction'] < outlier]
outlier = data['Age'].quantile(0.99)
data = data[data['Age'] < outlier]
pass
plotnumber = 1
for column in data:
    if plotnumber <= 9:
        pass
        pass
        pass
    plotnumber += 1
pass
corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
pass
X = data.drop(columns=['Outcome'])
y = data['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
lr = LogisticRegression()