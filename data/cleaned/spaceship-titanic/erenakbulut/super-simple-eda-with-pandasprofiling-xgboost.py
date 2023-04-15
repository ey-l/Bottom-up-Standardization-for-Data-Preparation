import pandas as pd
import numpy as np
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.head()
train.shape
train.columns
train.describe()
train.info()
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(train['Transported'])

corrmat = train.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data
import missingno as msno
msno.matrix(train)
import pandas_profiling as pp
report_train = pp.ProfileReport(train)
report_train
from sklearn.preprocessing import LabelEncoder
categorical_values = train.select_dtypes(include=['object']).columns
for i in categorical_values:
    lbl = LabelEncoder()