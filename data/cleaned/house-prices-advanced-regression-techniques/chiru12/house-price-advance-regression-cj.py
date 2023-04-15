import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
data_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data_train.head()
import missingno as msno
msno.matrix(data_train)
msno.heatmap(data_train)
corr = data_train.corr()
sns.heatmap(corr)
data_train.isnull().sum()
data_tr = data_train.copy()
data_tr.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
data_tr['LotFrontage'] = data_tr['LotFrontage'].fillna(data_tr['LotFrontage'].mean())
data_tr['MasVnrArea'] = data_tr['MasVnrArea'].fillna(data_tr['MasVnrArea'].mean())
data_tr['GarageYrBlt'] = data_tr['GarageYrBlt'].fillna(data_tr['GarageYrBlt'].mean())
df_most_common_imputed = data_tr.apply(lambda x: x.fillna(x.value_counts().index[0]))
df_most_common_imputed.dropna(axis=0, inplace=True)
df_most_common_imputed
df_most_common_imputed.isnull().sum()
import pandas as pd
import numpy as np
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in df_most_common_imputed.columns:
    if df_most_common_imputed[column_name].dtype == object:
        df_most_common_imputed[column_name] = le.fit_transform(df_most_common_imputed[column_name])
    else:
        pass
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = df_most_common_imputed.iloc[:, 0:]
y = df_most_common_imputed.iloc[:, -1]
bestfeatures = SelectKBest(score_func=chi2, k=46)