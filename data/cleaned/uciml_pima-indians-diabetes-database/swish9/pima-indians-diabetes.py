import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score
dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataset.head(10)
dataset.isnull().sum()
dataset.info()
dataset.describe()
dataset.nunique()
sns.countplot(x='Pregnancies', data=dataset, hue='Outcome', color='green')
sns.relplot(data=dataset, x='Pregnancies', y='Age', hue='Outcome')
palette = sns.cubehelix_palette(light=0.8, n_colors=6)
sns.relplot(data=dataset, kind='line', x='Age', y='Glucose', hue='Outcome', palette=palette)
y = dataset.Outcome
X = dataset.drop('Outcome', axis=1)
corr_matrix = dataset.corr()
sns.heatmap(corr_matrix, vmax=0.8, linewidths=0.01, square=True, annot=True, cmap='YlGnBu', linecolor='black')
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
model = LinearRegression()
rfe = RFE(model, n_features_to_select=6)