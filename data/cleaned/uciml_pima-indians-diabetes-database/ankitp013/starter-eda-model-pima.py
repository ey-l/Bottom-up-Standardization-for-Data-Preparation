import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly
import plotly.express as ex
import plotly.graph_objs as go
import ipywidgets
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from pandas_profiling import ProfileReport
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.columns
data.info()
data.describe()
data.isna().sum()
data.dtypes
file = ProfileReport(data)
file.to_file(output_file='output.html')
file
df1 = data.loc[data['Outcome'] == 1]
df2 = data.loc[data['Outcome'] == 0]
df1 = df1.replace({'BloodPressure': 0}, np.median(df1['BloodPressure']))
df2 = df2.replace({'BloodPressure': 0}, np.median(df2['BloodPressure']))
dataframe = [df1, df2]
data = pd.concat(dataframe)
fig = make_subplots(rows=2, cols=2)
fig.add_trace(go.Histogram(x=data['Glucose'], histnorm='probability', marker_color='#EB89B5', name='Glucose:Histogram'), row=1, col=1)
fig.add_trace(go.Violin(y=data['Glucose'], box_visible=True, meanline_visible=True, points='all', jitter=0.05, marker_color='yellow', name='Glucose:Violin'), row=1, col=2)
fig.add_trace(go.Histogram(x=data['BloodPressure'], histnorm='probability', marker_color='#EB89B5', name='BloodPressure:Histogram'), row=2, col=1)
fig.add_trace(go.Violin(y=data['BloodPressure'], box_visible=True, meanline_visible=True, points='all', jitter=0.05, marker_color='yellow', name='BloodPressure:Violin'), row=2, col=2)
fig.update_xaxes(title_text='Glucose : Histogram Distribution', row=1, col=1)
fig.update_xaxes(title_text='Glucose : Violin Distribution', row=1, col=2)
fig.update_xaxes(title_text='BloodPressure : Histogram Distribution', row=2, col=1)
fig.update_xaxes(title_text='BloodPressure : Violin Distribution', row=2, col=2)
fig.update_layout(title={'text': 'Distribution Plots:glucose and blood pressure', 'xanchor': 'center', 'yanchor': 'top', 'x': 0.5, 'y': 0.97}, template='plotly_dark', height=900, legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='center', x=0.5, bgcolor='black', bordercolor='white', borderwidth=2, font=dict(family='Courier', size=10, color='white')), annotations=[dict(showarrow=True, x=155, y=0.04, xref='x1', yref='y1', text='Slightly Right Skewed', xanchor='left', xshift=10, opacity=0.7, font=dict(color='red', size=12), arrowcolor='lightgoldenrodyellow', arrowsize=5, arrowwidth=0.5, arrowhead=4), dict(showarrow=True, x=39, y=0.06, xref='x3', yref='y3', text='Close to Normal distribution', xanchor='left', xshift=10, opacity=0.7, font=dict(color='red', size=12), arrowcolor='lightgoldenrodyellow', arrowsize=5, arrowwidth=0.5, arrowhead=4)])
fig.show()
col_y = 'Outcome'
targets = [data.loc[data[col_y] == val] for val in data[col_y].unique()]
fig = go.Figure()
fig.add_trace(go.Histogram(x=targets[0]['Glucose'], histnorm='probability', marker_color='yellow', name='Outcome:1'))
fig.add_trace(go.Histogram(x=targets[1]['Glucose'], histnorm='probability', marker_color='red', name='Outcome:0'))
fig.update_xaxes(title_text='Glucose : Histogram Distribution')
fig.update_layout(title={'text': 'Distribution :Glucose Hue on Outcome', 'xanchor': 'center', 'yanchor': 'top', 'x': 0.5, 'y': 0.97}, template='plotly_dark', legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='center', x=0.5, bgcolor='black', bordercolor='white', borderwidth=2, font=dict(family='Courier', size=10, color='white')), annotations=[dict(showarrow=True, x=65, y=0.06, text=' Outcome Distribution <br> for 0', xanchor='right', xshift=10, opacity=0.9, font=dict(color='lightgoldenrodyellow', size=12), arrowcolor='lightgoldenrodyellow', arrowsize=5, arrowwidth=0.5, arrowhead=4), dict(showarrow=True, x=139, y=0.12, text=' Outcome Distribution <br> for 1', xanchor='left', xshift=10, opacity=0.9, font=dict(color='lightgoldenrodyellow', size=12), arrowcolor='lightgoldenrodyellow', arrowsize=5, arrowwidth=0.5, arrowside='end', arrowhead=4)], barmode='overlay')
fig.update_traces(opacity=0.5)
fig.show()
fig = go.Figure()
fig.add_trace(go.Box(x=data['Outcome'], y=data['SkinThickness'], marker=dict(color='limegreen', outliercolor='red', line=dict(outliercolor='red', outlierwidth=2)), name='Outcome', jitter=0.5, boxmean='sd', boxpoints='suspectedoutliers'))
fig.update_xaxes(title_text='SkinThickness : Boxplot')
fig.update_layout(title={'text': 'Boxplot :SkinThickness Hue on Outcome', 'xanchor': 'center', 'yanchor': 'top', 'x': 0.5, 'y': 0.97}, template='plotly_dark', legend=dict(yanchor='bottom', y=0.5, xanchor='center', x=1.2), barmode='overlay', annotations=[dict(showarrow=True, x=1, y=100, text=' Outlier', xanchor='right', xshift=0, opacity=0.9, font=dict(color='lightgoldenrodyellow', size=12), arrowcolor='lightgoldenrodyellow', arrowsize=5, arrowwidth=0.5, arrowhead=4)])
fig.update_traces(opacity=0.5)
fig.show()
fig = ex.violin(data, y='Outcome', x='Insulin', color='Outcome', orientation='h').update_traces(side='positive', width=2)
fig.update_xaxes(title_text='Insulin : Violin Side Positive Hue on Outcome ')
fig.update_layout(template='plotly_dark', legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='center', x=0.5, bgcolor='black', bordercolor='white', borderwidth=2, font=dict(family='Courier', size=10, color='white')))
data.columns
from plotly.graph_objs import *
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
trace1 = {'type': 'heatmap', 'x': data[cols].corr().columns.tolist(), 'y': data[cols].corr().columns.tolist(), 'z': data[cols].corr().values.tolist()}
df = Data([trace1])
fig = Figure(data=df)
fig.update_layout(template='plotly_dark', title={'text': 'Features Correlation Matrix', 'xanchor': 'center', 'yanchor': 'top', 'x': 0.5, 'y': 0.9})
fig.show()
data['age_bucket'] = np.where(data['Age'] <= 30, '0-30', np.where((data['Age'] > 30) & (data['Age'] <= 40), '30-40', np.where((data['Age'] > 40) & (data['Age'] <= 50), '40-50', np.where((data['Age'] > 50) & (data['Age'] <= 60), '50-60', '60+'))))
data['Outcome'] = data['Outcome'].astype(object)
fig = ex.parallel_categories(data, dimensions=['age_bucket', 'Outcome'], labels={'age_bucket': 'Age Bucket', 'Outcome': 'Diabetes'}, template='ggplot2', title='Parallely Corordinates : Distribution of Age bucket vís-a-vía Diabetes')
fig.update_layout()
fig.show()
data = pd.get_dummies(data, columns=['Pregnancies'], drop_first=True)
data['Outcome'] = data['Outcome'].astype(int)
col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Pregnancies_1', 'Pregnancies_2', 'Pregnancies_3', 'Pregnancies_4', 'Pregnancies_5', 'Pregnancies_6', 'Pregnancies_7', 'Pregnancies_8', 'Pregnancies_9', 'Pregnancies_10', 'Pregnancies_11', 'Pregnancies_12', 'Pregnancies_13', 'Pregnancies_14', 'Pregnancies_15', 'Pregnancies_17']
X = data
Y = X['Outcome'].values
X = X.drop('Outcome', axis=1)
X = X[col]
(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.1)
import warnings
warnings.filterwarnings('ignore')
all_classifier = []
all_classifier.append(('LR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression())])))
all_classifier.append(('KNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier(n_neighbors=3))])))
all_classifier.append(('DT', Pipeline([('Scaler', StandardScaler()), ('DT', DecisionTreeClassifier(criterion='gini', max_depth=9, min_samples_leaf=10, random_state=42))])))
all_classifier.append(('RF', Pipeline([('Scaler', StandardScaler()), ('RF', RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=42))])))
all_classifier.append(('ADB', Pipeline([('Scaler', StandardScaler()), ('ADB', AdaBoostClassifier(n_estimators=500))])))
all_classifier.append(('BC', Pipeline([('Scaler', StandardScaler()), ('BC', BaggingClassifier(n_estimators=500))])))
train_acc = []
test_acc = []
train_acc2 = []
test_acc2 = []
for (name, model) in all_classifier:
    kfold = KFold(n_splits=5)
    cv_results_train = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    cv_results_test = cross_val_score(model, X_test, y_test, cv=kfold, scoring='accuracy')
    cv_results_train2 = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1')
    cv_results_test2 = cross_val_score(model, X_test, y_test, cv=kfold, scoring='f1')
    train_acc.append(cv_results_train.mean())
    test_acc.append(cv_results_test.mean())
    train_acc2.append(cv_results_train2.mean())
    test_acc2.append(cv_results_test2.mean())
col = {'Train Acc': train_acc, 'Test Acc': test_acc, 'Train F1': train_acc2, 'Test F1': test_acc2}
models = ['Logistic Regression', 'KNN', 'Decsion Tree', 'Random Forest', 'ADA Boost', 'Bagging']
rslt = pd.DataFrame(data=col, index=models)
rslt
fig = ex.bar(rslt, barmode='group', template='plotly_dark')
fig.update_layout(title={'text': 'Results of Baseline Models', 'xanchor': 'center', 'yanchor': 'top', 'x': 0.5, 'y': 0.96}, legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5, bgcolor='black', bordercolor='white', borderwidth=2, font=dict(family='Courier', size=10, color='white')), xaxis_title='Models', yaxis_title='Scores')