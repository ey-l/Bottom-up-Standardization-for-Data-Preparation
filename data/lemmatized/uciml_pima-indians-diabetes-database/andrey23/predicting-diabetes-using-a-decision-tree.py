import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn import tree
import graphviz
import pydot
from sklearn.metrics import accuracy_score
features = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
features.head(5)
print('The shape of our features is:', features.shape)
features.info()
features.describe()
from plotly.subplots import make_subplots
fig = make_subplots(rows=5, cols=2, start_cell='bottom-left')
fig.add_trace(go.Scatter(y=features['Pregnancies'], mode='markers'), row=1, col=1)
fig.add_trace(go.Scatter(y=features['Glucose'], mode='markers'), row=1, col=2)
fig.add_trace(go.Scatter(y=features['BloodPressure'], mode='markers'), row=2, col=1)
fig.add_trace(go.Scatter(y=features['SkinThickness'], mode='markers'), row=2, col=2)
fig.add_trace(go.Scatter(y=features['Insulin'], mode='markers'), row=3, col=1)
fig.add_trace(go.Scatter(y=features['BMI'], mode='markers'), row=3, col=2)
fig.add_trace(go.Scatter(y=features['DiabetesPedigreeFunction'], mode='markers'), row=4, col=1)
fig.add_trace(go.Scatter(y=features['Age'], mode='markers'), row=4, col=2)
fig.add_trace(go.Scatter(y=features['Outcome'], mode='markers'), row=5, col=1)
fig.update_yaxes(title_text='Pregnancies', row=1, col=1)
fig.update_yaxes(title_text='Glucose', row=1, col=2)
fig.update_yaxes(title_text='BloodPressure', row=2, col=1)
fig.update_yaxes(title_text='SkinThickness', row=2, col=2)
fig.update_yaxes(title_text='Insulin', row=3, col=1)
fig.update_yaxes(title_text='BMI', row=3, col=2)
fig.update_yaxes(title_text='DiabetesPedigreeFunction', row=4, col=1)
fig.update_yaxes(title_text='Age', row=4, col=2)
fig.update_yaxes(title_text='Outcome', row=5, col=1)
fig.update_layout(title_text='Basic plots for data verifying and detecting outliers', height=1400, width=1000, showlegend=False)
pass
features.isna().sum()
(features == 0).sum()
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    features[col] = features[col].mask(features[col] == 0, features[col].median())
(features == 0).sum()
target = features['Outcome']
features = features.drop('Outcome', axis=1)
list(features.columns)

def detect_outliers_zscore(dataframe, col_name):
    outliers = []
    thres = 3
    mean = np.mean(dataframe[col_name])
    std = np.std(dataframe[col_name])
    for i in dataframe[col_name]:
        z_score = (i - mean) / std
        if np.abs(z_score) > thres:
            outliers.append(i)
    return outliers
print(detect_outliers_zscore(features, 'Pregnancies'))
print(detect_outliers_zscore(features, 'Glucose'))
print(detect_outliers_zscore(features, 'BloodPressure'))
print(detect_outliers_zscore(features, 'SkinThickness'))
print(detect_outliers_zscore(features, 'Insulin'))
print(detect_outliers_zscore(features, 'BMI'))
print(detect_outliers_zscore(features, 'DiabetesPedigreeFunction'))
cols_with_outliers = [features['Pregnancies'], features['BloodPressure'], features['SkinThickness'], features['Insulin'], features['BMI'], features['DiabetesPedigreeFunction']]

def replace_outliers(dataframe, col_name):
    for i in detect_outliers_zscore(dataframe, col_name):
        dataframe.loc[dataframe[col_name] == i] = dataframe[col_name].median()
replace_outliers(features, 'Pregnancies')
replace_outliers(features, 'BloodPressure')
replace_outliers(features, 'SkinThickness')
replace_outliers(features, 'Insulin')
replace_outliers(features, 'BMI')
replace_outliers(features, 'DiabetesPedigreeFunction')
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(features, target, test_size=0.25, random_state=42)
print('Training Features Shape:', X_train.shape)
print('Training Target Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Target Shape:', y_test.shape)
dtc = tree.DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=3, min_samples_leaf=4)