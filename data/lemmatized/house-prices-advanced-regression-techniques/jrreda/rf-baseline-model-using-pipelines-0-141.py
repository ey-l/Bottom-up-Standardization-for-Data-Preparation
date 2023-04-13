import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=14, titlepad=10)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.info()
_input1.isna().sum().sum()
_input1[_input1.columns[_input1.isna().any()]].isna().sum()
_input1.isna().sum().plot(kind='hist')
plt.xlabel('Number of NaNs')
plt.title('NaNs distribution', fontsize=18)
threshold = 800
print(f"Columns with grater than {threshold} NaNs, {round(threshold / _input1.shape[0] * 100)}% of it's values are NaNs.")
outliers = _input1.isna().sum()[_input1.isna().sum() > threshold]
outliers
_input1.duplicated().sum()
_input1.describe()
_input1.hist(bins=50, figsize=(20, 16))
sns.heatmap(_input1.corr(), annot=False, cmap='icefire')
_input1.corr().unstack().sort_values(ascending=False).drop_duplicates()[:24]
_input1.corrwith(_input1['SalePrice']).sort_values(ascending=False)
_input1.corrwith(_input1['SalePrice']).sort_values(ascending=False)[1:].plot(kind='bar')
plt.vlines(26.5, -0.1, 0.8, colors='red')
plt.title("Correlations with the Target 'SalePrice'")
X = _input1.drop(columns=['SalePrice'])
y = _input1.SalePrice
less_than_0_corr = _input1.corrwith(_input1['SalePrice'])[_input1.corrwith(_input1['SalePrice']) < 0].index.to_list()
cols_to_remove = list(outliers.index) + less_than_0_corr
num_df = X.select_dtypes(include='number')
num_df.head()
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)), ('scaler', MinMaxScaler())])
cat_df = X.select_dtypes(include='object')
cat_df.head()
from sklearn.preprocessing import OrdinalEncoder
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)), ('encoder', OrdinalEncoder())])
from sklearn.compose import ColumnTransformer
num_attribs = list(set(num_df) - set(cols_to_remove))
cat_attribs = list(set(cat_df) - set(cols_to_remove))
preprocessor = ColumnTransformer([('num', num_pipeline, num_attribs), ('cat', cat_pipeline, cat_attribs)], remainder='drop')
X_prepared = preprocessor.fit_transform(X)
X_prepared.shape
random_state = 10
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
log_y = np.log(y)