import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
data1 = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data1
data1.info()
data1.columns
data1.isnull().any()
data1.isnull().sum()
data1.duplicated().any()
data1.describe()
pass
pass
q1 = data1['Glucose'].quantile(0.25)
q3 = data1['Glucose'].quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR
data1[data1['Glucose'] > upper_limit]
data1 = data1[(data1['Glucose'] > lower_limit) & (data1['Glucose'] < upper_limit)]
pass
q1 = data1['BloodPressure'].quantile(0.25)
q3 = data1['BloodPressure'].quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR
data1[data1['BloodPressure'] > upper_limit]
data1 = data1[(data1['BloodPressure'] > lower_limit) & (data1['BloodPressure'] < upper_limit)]
pass
q1 = data1['SkinThickness'].quantile(0.25)
q3 = data1['SkinThickness'].quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR
data1[data1['SkinThickness'] > upper_limit]
data1 = data1[(data1['SkinThickness'] > lower_limit) & (data1['SkinThickness'] < upper_limit)]
pass
q1 = data1['Insulin'].quantile(0.25)
q3 = data1['Insulin'].quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR
data1[data1['Insulin'] > upper_limit]
data1 = data1[(data1['Insulin'] > lower_limit) & (data1['Insulin'] < upper_limit)]
pass
q1 = data1['BMI'].quantile(0.25)
q3 = data1['BMI'].quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR
data1[data1['BMI'] > upper_limit]
data1 = data1[(data1['BMI'] > lower_limit) & (data1['BMI'] < upper_limit)]
pass
q1 = data1['Age'].quantile(0.25)
q3 = data1['Age'].quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR
data1[data1['Age'] > upper_limit]
data1 = data1[(data1['Age'] > lower_limit) & (data1['Age'] < upper_limit)]
pass
pass
pass
data1.describe()
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
data1.hist(figsize=(20, 15), bins=30)
x = data1.drop('Outcome', axis=1)
y = data1['Outcome']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=130)
(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
MODEL = []
TEST = []
TRAIN = []

def result(model, test, train):
    MODEL.append(model)
    TEST.append(round(test, 2))
    TRAIN.append(round(train, 2))
from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression()