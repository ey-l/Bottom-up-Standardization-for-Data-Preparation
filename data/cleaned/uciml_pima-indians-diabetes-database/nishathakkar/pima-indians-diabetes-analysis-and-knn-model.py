import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataset.head()
(x_train, x_test, y_train, y_test) = train_test_split(dataset.drop('Outcome', axis=1), dataset['Outcome'], test_size=0.3, random_state=0)
(x_train.shape, x_test.shape)
x_train.dtypes
discrete_variables = [var for var in x_train.columns if x_train[var].dtype != 'O' and x_train[var].nunique() < 10]
continuous_variables = [var for var in x_train.columns if var not in discrete_variables and x_train[var].dtype != 'O']
discrete_variables
continuous_variables
x_train.isnull().sum()
x_test.isnull().sum()

def diagnostic_plot(df, var):
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    df[var].plot(kind='hist', bins=50)
    plt.title('Histogram')
    plt.xlabel(var)
    plt.subplot(1, 3, 2)
    stats.probplot(df[var], dist='norm', plot=plt)
    plt.ylabel('RM quanriles')
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[var])
    plt.title('Boxplot')

plt.style.use('dark_background')
for var in continuous_variables:
    diagnostic_plot(x_train, var)
correlation_matrix = np.round(x_train.corr(), 2)
fig = plt.figure(figsize=(10, 10))
sns.heatmap(data=correlation_matrix, annot=True)
sns.lmplot(x='Age', y='Pregnancies', data=x_train)

sns.lmplot(x='Age', y='SkinThickness', data=x_train)

x_train.describe()

from feature_engine.discretisation import EqualWidthDiscretiser
disc = EqualWidthDiscretiser(variables=continuous_variables, bins=5, return_object=True)