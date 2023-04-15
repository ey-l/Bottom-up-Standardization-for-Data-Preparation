import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import matplotlib as mpl
import matplotlib.patches as mpatches
import plotly.express as px
from plotly import tools
from plotly.subplots import make_subplots
from plotly.offline import iplot

sns.set(rc={'figure.figsize': [10, 10]}, font_scale=1.2)
from numpy import sqrt, abs, round
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score, classification_report
df_original = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df = df_original.copy()
df.shape
df.sample(10)
df.info()
df.describe()
dict_ = {}
for i in df.columns:
    dict_[i] = df[i].value_counts().shape[0]
pd.DataFrame(dict_, index=['Unique']).T
print(f'Number of duplicated rows is: {df.duplicated().sum()}')

def distripution(feature):
    sns.histplot(df, x=feature, color=sns.color_palette()[4])
    plt.title('Distribution Of The' + feature + 'Feature', fontsize=15, loc='center', fontweight=10, fontstyle='italic')
    plt.ylabel('Frequency', fontsize=12.5)
    plt.xlabel(feature, fontsize=12.5)
for i in df.columns[:-1]:
    distripution(i)

sns.countplot(x='Outcome', data=df, palette='mako')
plt.title('Distribution Of The Target Feature (Outcome)', fontsize=15, loc='center', fontweight=10, fontstyle='italic')
plt.ylabel('Frequency', fontsize=12.5)
plt.xlabel('Outcome', fontsize=12.5)
from imblearn.over_sampling import SMOTE
(x, y) = (df.drop('Outcome', axis=1), df['Outcome'])
smote = SMOTE()
(x, y) = smote.fit_resample(x, y)
df = pd.concat([x, y], axis=1)
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].mean())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())
df['Glucose'] = df['Glucose'].astype(int)
df['BloodPressure'] = df['BloodPressure'].astype(int)
df['SkinThickness'] = df['SkinThickness'].astype(int)
df['Insulin'] = df['Insulin'].astype(int)
df['Pregnancies'] = df['Pregnancies'].astype(int)
df['Age'] = df['Age'].astype(int)
df['Outcome'] = df['Outcome'].astype(int)
df.info()
for i in df.columns[:-1]:
    distripution(i)


def Impact_visuals(feature):
    fig = px.histogram(df, x=feature, color='Outcome', hover_data=df.columns, template='simple_white', color_discrete_sequence=px.colors.diverging.Earth_r)
    fig.update_layout(title_text='<b> How does' + feature + ' impact on the outcome?</b>', title_x=0.5, font_size=15)
    fig.show()
for i in df.columns[:-1]:
    Impact_visuals(i)

sns.pairplot(df, hue='Outcome', palette='mako_r')
fig = plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(), annot=True, cmap='mako_r', linewidths=2, linecolor='white', cbar=True)
cols_names = list(df.corr().columns)[:-1]
cols_values = list(df.corr().Outcome)
dict__ = {}
count = 0
for i in cols_names[:-1]:
    dict__[i] = cols_values[count]
    count += 1
data = dict__
Features = list(data.keys())
values = list(data.values())
fig = plt.figure(figsize=(15, 5))
plt.bar(Features, values, color='maroon', width=0.4)
plt.xlabel('Features')
plt.title('correlation between features and Outcome')
corr = df.corr(method='kendall')
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap='YlGnBu', annot=True)
df_kendall = df[['Glucose', 'BMI', 'Age', 'Outcome']]
df_kendall.head()
(X, y) = (df_kendall.drop('Outcome', axis=1), df['Outcome'])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.15, random_state=0)
scaler = StandardScaler()