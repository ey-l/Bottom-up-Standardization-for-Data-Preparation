import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
pima = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
pima.head()
pima.info()
pass
pass
pima['Outcome'].value_counts()
pass
pima.describe()
corrmat = pima.corr()
pass
pass
pima.columns
pima.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(10, 8))
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pima_new = pima
pima_new.info()
pima_new = pima_new[pima_new['Pregnancies'] < 13]
pima_new = pima_new[pima_new['Glucose'] > 30]
pima_new = pima_new[pima_new['BMI'] > 10]
pima_new = pima_new[pima_new['BMI'] < 50]
pima_new = pima_new[pima_new['DiabetesPedigreeFunction'] < 1.2]
pima_new = pima_new[pima_new['Age'] < 65]
pima_new.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(10, 8))
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(pima_new.drop('Outcome', axis=1), pima_new['Outcome'], test_size=0.3, random_state=123)
from sklearn import preprocessing