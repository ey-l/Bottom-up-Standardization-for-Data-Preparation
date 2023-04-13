import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pima = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
pima.head()
(pima.shape, pima.keys(), type(pima))
pima.describe()
pima.groupby('Outcome').size()
pima.hist(figsize=(16, 14))
pass
pass
pass
pima.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(16, 14))
pima.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(16, 14))
c = pima.iloc[:, 0:8].corr()
c
corr = pima.corr()
corr
pass
pass
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = pima.iloc[:, 0:8]
Y = pima.iloc[:, 8]
select_top_4 = SelectKBest(score_func=chi2, k=4)
(X.shape, pima.shape)
X.head()