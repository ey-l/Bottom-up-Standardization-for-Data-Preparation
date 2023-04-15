import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
dataframe = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df = dataframe.copy()
df.head()
print('The columns are:', df.columns)
print('The shape of the dataframe is:', df.shape)
df.describe()
df.info()
for column in df.columns:
    print(column, 'has', df[column].nunique(), ' number of unique categories.')
plt.figure(figsize=(20, 20))
for tups in enumerate(df.columns):
    plt.subplot(3, 3, tups[0] + 1)
    sns.set(rc={'figure.figsize': (7, 5)})
    sns.histplot(data=df, x=tups[1], kde=True)
    plt.title("{}'s distribuition.".format(tups[1]))

independent_feature = [column for column in df.columns if column not in ['Outcome']]
plt.figure(figsize=(20, 20))
for tups in enumerate(independent_feature):
    plt.subplot(3, 3, tups[0] + 1)
    sns.set(rc={'figure.figsize': (7, 5)})
    sns.boxplot(data=df, x=tups[1])
    plt.title('{}'.format(tups[1]))


def outlier_trimmer(data_set, feature, trimming_value, pos='upper'):
    threshold = data_set[feature].quantile(trimming_value / 100)
    if pos == 'lower':
        data_set = data_set[data_set[feature] > threshold]
    else:
        data_set = data_set[data_set[feature] < threshold]
    return data_set
for column in ['BloodPressure', 'SkinThickness', 'BMI', 'Age']:
    df = outlier_trimmer(df, column, 99)
print('Shape of dataframe after trimming BloodPressure, SkinThickness, BMI and age is: ', df.shape)
for column in ['Insulin', 'DiabetesPedigreeFunction']:
    df = outlier_trimmer(df, column, 97)
print('Shape of dataframe after trimming Insulin and DiabetesPedigreeFunction is: ', df.shape)
for column in ['BMI', 'Glucose', 'BloodPressure']:
    df = outlier_trimmer(df, column, 0.5, 'lower')
print('Shape of dataframe after trimming BloodPressure, Glucose and BMI is: ', df.shape)
plt.figure(figsize=(20, 20))
for tups in enumerate(independent_feature):
    plt.subplot(3, 3, tups[0] + 1)
    sns.set(rc={'figure.figsize': (7, 5)})
    sns.boxplot(data=df, x=tups[1])
    plt.title('{}'.format(tups[1]))

plt.figure(figsize=(20, 20))
for tups in enumerate(df.columns):
    plt.subplot(3, 3, tups[0] + 1)
    sns.set(rc={'figure.figsize': (7, 5)})
    sns.histplot(data=df, x=tups[1], kde=True)
    plt.title("{}'s distribuition.".format(tups[1]))

print('The count of negative and postive outputs in the original data are: \n', dataframe['Outcome'].value_counts())
sns.countplot(data=dataframe, x='Outcome')

print('The count of negative and postive outputs in the cleaned data are: \n', dataframe['Outcome'].value_counts())
sns.countplot(data=df, x='Outcome')

corr = df.corr(method='pearson')
sns.set(rc={'figure.figsize': (20, 10)})
sns.heatmap(data=corr, annot=True, linewidths=0.5, cmap='coolwarm', vmin=-1, vmax=1)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=33)
print('The shape of training data: ', X_train.shape, y_train.shape)
print('The shape of test data: ', X_test.shape, y_test.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
logistic_reg = LogisticRegression()