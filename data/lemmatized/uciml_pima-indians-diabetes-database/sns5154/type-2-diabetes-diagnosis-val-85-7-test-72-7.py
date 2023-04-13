import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config
set_config(display='diagram')
import warnings
warnings.filterwarnings('ignore')
from matplotlib import rcParams
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(data.shape)
data.head()
data.isnull().sum()
from sklearn.model_selection import train_test_split
(data, test) = train_test_split(data, test_size=0.1, random_state=42)
data.shape
data.info()
data.describe()
data.columns
pass
for i in enumerate(data.columns[:-1]):
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
for col in data.columns[:-1]:
    print(f'Statistical Maximum for {col}: {data[col].quantile(0.75) + 1.5 * (data[col].quantile(0.75) - data[col].quantile(0.25))}')
    print(f'Statistical Minimum for {col}: {data[col].quantile(0.25) - 1.5 * (data[col].quantile(0.75) - data[col].quantile(0.25))}')
    print()
for (idx, row) in data.iterrows():
    if data.loc[idx, 'Pregnancies'] > 13:
        data.loc[idx, 'Pregnancies'] = 13
for (idx, row) in data.iterrows():
    if data.loc[idx, 'Glucose'] > 200:
        data.loc[idx, 'Glucose'] = 200
    if data.loc[idx, 'Glucose'] < 40:
        data.loc[idx, 'Glucose'] = 40
for (idx, row) in data.iterrows():
    if data.loc[idx, 'BloodPressure'] > 107:
        data.loc[idx, 'BloodPressure'] = 107
    if data.loc[idx, 'BloodPressure'] < 35:
        data.loc[idx, 'BloodPressure'] = 35
for (idx, row) in data.iterrows():
    if data.loc[idx, 'Insulin'] > 323:
        data.loc[idx, 'Insulin'] = 323
for (idx, row) in data.iterrows():
    if data.loc[idx, 'BMI'] > 50:
        data.loc[idx, 'BMI'] = 50
    if data.loc[idx, 'BMI'] < 14:
        data.loc[idx, 'BMI'] = 14
for (idx, row) in data.iterrows():
    if data.loc[idx, 'DiabetesPedigreeFunction'] > 1.19:
        data.loc[idx, 'DiabetesPedigreeFunction'] = 1.19
for (idx, row) in data.iterrows():
    if data.loc[idx, 'Age'] > 64:
        data.loc[idx, 'Age'] = 64
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
tempData = data.copy()
tempData = tempData.drop('Outcome', axis=1)
tempData = sc.fit_transform(tempData)
temp_df = pd.DataFrame(tempData, columns=data.columns[:-1])
pass
for i in enumerate(data.columns[:-1]):
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    stats.probplot(temp_df[i[1]], dist='norm', plot=plt)
    pass
    pass
    pass
pass
pass
pass
from sklearn.ensemble import ExtraTreesClassifier
pass
model = ExtraTreesClassifier()