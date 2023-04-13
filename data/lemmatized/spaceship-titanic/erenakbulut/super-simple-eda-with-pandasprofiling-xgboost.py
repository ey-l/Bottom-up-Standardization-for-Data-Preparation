import pandas as pd
import numpy as np
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.shape
_input1.columns
_input1.describe()
_input1.info()
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(_input1['Transported'])
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data
import missingno as msno
msno.matrix(_input1)
import pandas_profiling as pp
report_train = pp.ProfileReport(_input1)
report_train
from sklearn.preprocessing import LabelEncoder
categorical_values = _input1.select_dtypes(include=['object']).columns
for i in categorical_values:
    lbl = LabelEncoder()