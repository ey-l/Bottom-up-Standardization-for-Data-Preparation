import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
print('Libraries imported')
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print('The shape of our dataset is', data.shape)
print('Number of columns', data.shape[1])
print('Number of rows', data.shape[0])
data.info()
data.head(5)
data.tail(5)
data['Outcome'].value_counts()
data.describe(include='all')
print(data.isna().sum().sort_values(ascending=False))
print('duplicate values in our data are ', data.duplicated().sum())
print('duplicated dropped')
data.columns
data.nunique()
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
IQR
((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).any()
data2 = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
data[data2]
data[data2].plot(kind='box', subplots=True, sharex=False, sharey=False, layout=(3, 3), figsize=(10, 10))


def find_outliers_IQR(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    outliers = df[(df < q1 - 1.5 * IQR) | (df > q3 + 1.5 * IQR)]
    return outliers
data.hist(data2, bins=25, figsize=(20, 15))

print(data.Pregnancies.value_counts())
sizes = data.Pregnancies.value_counts().values
labels = data.Pregnancies.value_counts().index
colors = ['green', 'pink', 'yellow', 'purple', 'grey', 'red', 'blue', 'darkblue', 'cyan', 'white', 'black']
plt.pie(sizes, radius=2, data=data, labels=labels, colors=colors)

sns.countplot(x='Outcome', data=data)
sns.heatmap(data[data2].corr(), annot=True)
data[data2].corr()
from sklearn.model_selection import train_test_split
y = data['Outcome']
X = data.drop(['Outcome'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.15, random_state=123)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.preprocessing import MinMaxScaler
SC = MinMaxScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import recall_score, classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, auc, f1_score
logistic_model = LogisticRegression()