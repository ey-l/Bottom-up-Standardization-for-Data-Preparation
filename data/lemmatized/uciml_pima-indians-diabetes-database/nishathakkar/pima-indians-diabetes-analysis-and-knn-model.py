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
    pass
    pass
    df[var].plot(kind='hist', bins=50)
    pass
    pass
    pass
    stats.probplot(df[var], dist='norm', plot=plt)
    pass
    pass
    pass
    pass
pass
for var in continuous_variables:
    diagnostic_plot(x_train, var)
correlation_matrix = np.round(x_train.corr(), 2)
pass
pass
pass
pass
x_train.describe()
from feature_engine.discretisation import EqualWidthDiscretiser
disc = EqualWidthDiscretiser(variables=continuous_variables, bins=5, return_object=True)