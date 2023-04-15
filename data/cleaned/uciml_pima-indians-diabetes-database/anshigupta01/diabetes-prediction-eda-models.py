import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
sns.set()

data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head(10)
data.shape
data.info()
data.describe()
data.isnull().sum()
import missingno as msno
msno.bar(data)

sns.heatmap(data.corr(), cbar=False, cmap='BuGn', annot=True)
col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for i in col:
    data[i].replace(0, data[i].mean(), inplace=True)
p = data.hist(figsize=(20, 20))
sns.pointplot(x='Outcome', y='Age', data=data)
sns.scatterplot(x='Age', y='Insulin', data=data)
sns.boxplot(x='Outcome', y='Pregnancies', data=data)
sns.pairplot(data, hue='Outcome')
sns.stripplot(x='Pregnancies', y='Age', data=data)
sns.regplot(x='SkinThickness', y='Insulin', data=data)
(f, ax) = plt.subplots(figsize=(10, 10))
ax = sns.swarmplot(x='Pregnancies', y='Age', hue='Outcome', palette='Dark2', data=data)
ax = sns.set(style='darkgrid')
g = sns.regplot(x='Insulin', y='Age', data=data)
data.var()
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(data.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = data.Outcome
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(X, y, test_size=0.3, random_state=3)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C=1, penalty='l2')