import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.io as pio
pio.renderers.default = 'svg'
py.init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import skew
from scipy.stats import kurtosis
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df_name = df.columns
print('dimension of data', df.shape)
df.info()
df.head()
df.describe()
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')

df.hist(bins=20, figsize=(18, 12))

for i in range(len(df.columns)):
    sns.kdeplot(df[df_name[i]], shade=True)

    print('%s: mean (%f), variance (%f), skewness (%f), kurtosis (%f)' % (df_name[i], np.mean(df[df_name[i]]), np.var(df[df_name[i]]), skew(df[df_name[i]]), kurtosis(df[df_name[i]])))
df.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False, fontsize=8, figsize=(18, 12))

print(df.groupby('Outcome').size())
OutLabels = [str(df['Outcome'].unique()[i]) for i in range(df['Outcome'].nunique())]
OutValues = [df['Outcome'].value_counts()[i] for i in range(df['Outcome'].nunique())]
pie = go.Pie(labels=OutLabels, values=OutValues)
go.Figure([pie])
sns.pairplot(df, hue='Outcome', palette='husl')

(fig, ax) = plt.subplots(figsize=(18, 12))
ax = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, fmt='.2f', cmap='coolwarm')

X = df[df_name[0:8]]
Y = df[df_name[8]]
(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=df['Outcome'])

def GetScaledModel(nameOfScaler):
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler == 'minmax':
        scaler = MinMaxScaler()
    pipelines = []
    pipelines.append((nameOfScaler + 'LR', Pipeline([('Scaler', scaler), ('LR', LogisticRegression())])))
    pipelines.append((nameOfScaler + 'LDA', Pipeline([('Scaler', scaler), ('LDA', LinearDiscriminantAnalysis())])))
    pipelines.append((nameOfScaler + 'KNN', Pipeline([('Scaler', scaler), ('KNN', KNeighborsClassifier())])))
    pipelines.append((nameOfScaler + 'CART', Pipeline([('Scaler', scaler), ('CART', DecisionTreeClassifier(random_state=2))])))
    pipelines.append((nameOfScaler + 'NB', Pipeline([('Scaler', scaler), ('NB', GaussianNB())])))
    pipelines.append((nameOfScaler + 'SVM', Pipeline([('Scaler', scaler), ('SVM', SVC())])))
    pipelines.append((nameOfScaler + 'AB', Pipeline([('Scaler', scaler), ('AB', AdaBoostClassifier())])))
    pipelines.append((nameOfScaler + 'GBM', Pipeline([('Scaler', scaler), ('GMB', GradientBoostingClassifier(random_state=2))])))
    pipelines.append((nameOfScaler + 'RF', Pipeline([('Scaler', scaler), ('RF', RandomForestClassifier(random_state=2))])))
    pipelines.append((nameOfScaler + 'ET', Pipeline([('Scaler', scaler), ('ET', ExtraTreesClassifier(random_state=2))])))
    return pipelines
num_folds = 10
seed = 7
scoring = 'accuracy'

def EvaluateAlg(X, y, nameOfScaler):
    results = []
    names = []
    models = GetScaledModel(nameOfScaler)
    for (name, model) in models:
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
    return (results, names)
(standard_results, standard_names) = EvaluateAlg(X_train, y_train, 'standard')
(minmax_results, minmax_names) = EvaluateAlg(X_train, y_train, 'minmax')
score = pd.DataFrame({'Model': standard_names, 'Score-mean': [np.mean(i) for i in standard_results], 'Model_2': minmax_names, 'Score-mean_2': [np.mean(i) for i in minmax_results]})
print(score)

def CompAlg(results, names):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 2, 2])
    ax.boxplot(results, labels=names, showmeans=True, meanline=True, meanprops=dict(linestyle='--', linewidth=2.5, color='green'))
    ax.yaxis.grid(True)
    ax.set_title('Algorithm Comparison')
    fig.text(1.8, 1.9, 'mean : ---', color='green', weight='roman', size=14)

CompAlg(standard_results, standard_names)
CompAlg(minmax_results, minmax_names)
df_copy = df.copy()

def OutliersBox(df, nameOfFeature):
    trace0 = go.Box(y=df[nameOfFeature], name='All Points', jitter=0.3, pointpos=-1.8, boxpoints='all', marker=dict(color='rgb(7,40,89)'), line=dict(color='rgb(7,40,89)'))
    trace1 = go.Box(y=df[nameOfFeature], name='Only Whiskers', boxpoints=False, marker=dict(color='rgb(9,56,125)'), line=dict(color='rgb(9,56,125)'))
    trace2 = go.Box(y=df[nameOfFeature], name='Suspected Outliers', boxpoints='suspectedoutliers', marker=dict(color='rgb(8,81,156)', outliercolor='rgba(219, 64, 82, 0.6)', line=dict(outliercolor='rgba(219, 64, 82, 0.6)', outlierwidth=2)), line=dict(color='rgb(8,81,156)'))
    data = [trace0, trace1, trace2]
    layout = go.Layout(title='{} Outliers'.format(nameOfFeature))
    go.Figure(data=data, layout=layout).show()

def DropOutliers(df_copy, nameOfFeature):
    valueOfFeature = df_copy[nameOfFeature]
    Q1 = np.percentile(valueOfFeature, 25.0)
    Q3 = np.percentile(valueOfFeature, 75.0)
    step = (Q3 - Q1) * 1.5
    outliers_idx = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].index.tolist()
    outliers_val = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].values
    print('Number of outliers (inc duplicates): {} and outliers: {}'.format(len(outliers_idx), outliers_val))
    good_data = df_copy.drop(df_copy.index[outliers_idx]).reset_index(drop=True)
    print('New dataset with removed outliers has {} samples with {} features each.'.format(*good_data.shape))
    return good_data
outliers_clmn = ['BloodPressure', 'DiabetesPedigreeFunction', 'Insulin', 'BMI', 'Glucose', 'SkinThickness']
df_clean = df_copy
for i in range(len(outliers_clmn)):
    OutliersBox(df, outliers_clmn[i])
    df_clean = DropOutliers(df_clean, outliers_clmn[i])
    OutliersBox(df_clean, outliers_clmn[i])
df_clean_name = df_clean.columns
X_c = df_clean[df_clean_name[0:8]]
Y_c = df_clean[df_clean_name[8]]
(X_train_c, X_test_c, y_train_c, y_test_c) = train_test_split(X_c, Y_c, test_size=0.2, random_state=0, stratify=df_clean['Outcome'])
(standard_results_c, standard_names_c) = EvaluateAlg(X_train_c, y_train_c, 'standard')
(minmax_results_c, minmax_names_c) = EvaluateAlg(X_train_c, y_train_c, 'minmax')
score_c = pd.DataFrame({'Model-s_c': standard_names_c, 'Score-s_c': [np.mean(i) for i in standard_results_c], 'Model-m_c': minmax_names_c, 'Score-m_c': [np.mean(i) for i in minmax_results_c]})
score = pd.concat([score, score_c], axis=1)
score
CompAlg(standard_results_c, standard_names_c)
CompAlg(minmax_results_c, minmax_names_c)
clf = ExtraTreesClassifier(n_estimators=250, random_state=2)