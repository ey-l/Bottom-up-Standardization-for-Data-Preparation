import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.shape
data.shape[0]
data.shape[1]
data.head()
data.tail()
data.info()
data.dtypes
data2 = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
data[data2].corr()
data.isna().sum()
data.duplicated().sum()
data.describe()
data.describe(include='all')
data.columns
data.nunique()
data.Age.plot(color='blue', kind='hist')
pass
pass
data.Glucose.plot(color='green', kind='hist')
pass
pass
pass
pass
data.Outcome.value_counts()
sizes = data.Pregnancies.value_counts().values
labels = data.Pregnancies.value_counts().index
colors = ['green', 'pink', 'yellow', 'purple', 'grey', 'red', 'blue', 'darkblue', 'cyan', 'white', 'black']
pass
data2 = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
data.hist(data2, bins=50, figsize=(20, 15))
pass
data.describe()
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
IQR
((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).any()

def find_outliers_IQR(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    IQR = q3 - q1
    outliers = data[(data < q1 - 1.5 * IQR) | (data > q3 + 1.5 * IQR)]
    return outliers
outliers = find_outliers_IQR(data['BloodPressure'])
print('number of outliers: ' + str(len(outliers)))
print('max outlier value: ' + str(outliers.max()))
print('min outlier value: ' + str(outliers.min()))
outliers
data[data2].plot(kind='box', subplots=True, sharex=False, sharey=False, layout=(3, 3), figsize=(10, 10))
pass
from sklearn.model_selection import train_test_split
X = data.drop(['Outcome'], axis=1)
y = data['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=123)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.preprocessing import MinMaxScaler
SC = MinMaxScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import recall_score, classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, auc, f1_score
logistic_model = LogisticRegression()