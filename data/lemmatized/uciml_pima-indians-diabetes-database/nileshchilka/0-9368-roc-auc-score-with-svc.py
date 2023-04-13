import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import scipy.stats as stats
from matplotlib import pylab
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df['Outcome'].value_counts()
df.isnull().sum()

def diagnostic_plots(df, variable):
    pass
    pass
    df[variable].hist()
    pass
    stats.probplot(df[variable], dist='norm', plot=pylab)
diagnostic_plots(df, 'Pregnancies')
df['Pregnancies'] = df.Pregnancies ** (1 / 1.4)
diagnostic_plots(df, 'Pregnancies')
diagnostic_plots(df, 'Glucose')
df['Glucose'] = df.Glucose ** 0.95
diagnostic_plots(df, 'Glucose')
diagnostic_plots(df, 'BloodPressure')
df['BloodPressure'] = df.BloodPressure ** 1.4
diagnostic_plots(df, 'BloodPressure')
diagnostic_plots(df, 'SkinThickness')
diagnostic_plots(df, 'Insulin')
df['Insulin'] = df.Insulin ** 0.4
diagnostic_plots(df, 'Insulin')
diagnostic_plots(df, 'BMI')
diagnostic_plots(df, 'DiabetesPedigreeFunction')
df['DiabetesPedigreeFunction'] = df.DiabetesPedigreeFunction ** 0.1
diagnostic_plots(df, 'DiabetesPedigreeFunction')
diagnostic_plots(df, 'Age')
df['Age'] = np.log(df.Age)
diagnostic_plots(df, 'Age')
for feature in df.columns[:-1]:
    IQR = df[feature].quantile(0.75) - df[feature].quantile(0.25)
    upper_bond = df[feature].quantile(0.75) + IQR * 1.5
    lower_bond = df[feature].quantile(0.25) - IQR * 1.5
    df[feature] = np.where(df[feature] > upper_bond, upper_bond, df[feature])
    df[feature] = np.where(df[feature] < lower_bond, lower_bond, df[feature])
for feature in df.columns[:-1]:
    df[f'{feature}_zero'] = np.where(df[feature] == 0, 1, 0)
    df[feature] = np.where((df[feature] == 0) & (df['Outcome'] == 0), df.groupby('Outcome')[feature].median()[0], df[feature])
    df[feature] = np.where((df[feature] == 0) & (df['Outcome'] == 1), df.groupby('Outcome')[feature].median()[1], df[feature])
X = df.drop('Outcome', axis=1)
X = StandardScaler().fit_transform(X)
y = df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=0)
y_train.value_counts()
y_test.value_counts()
model = SVC()
parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5, scoring='roc_auc')