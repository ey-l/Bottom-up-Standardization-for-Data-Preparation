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
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train_df.head()
train_df.info()
train_df.isna().sum().sum()
train_df[train_df.columns[train_df.isna().any()]].isna().sum()
train_df.isna().sum().plot(kind='hist')
plt.xlabel('Number of NaNs')
plt.title('NaNs distribution', fontsize=18)
threshold = 800
print(f"Columns with grater than {threshold} NaNs, {round(threshold / train_df.shape[0] * 100)}% of it's values are NaNs.")
outliers = train_df.isna().sum()[train_df.isna().sum() > threshold]
outliers
train_df.duplicated().sum()
train_df.describe()
train_df.hist(bins=50, figsize=(20, 16))
sns.heatmap(train_df.corr(), annot=False, cmap='icefire')
train_df.corr().unstack().sort_values(ascending=False).drop_duplicates()[:24]
train_df.corrwith(train_df['SalePrice']).sort_values(ascending=False)
train_df.corrwith(train_df['SalePrice']).sort_values(ascending=False)[1:].plot(kind='bar')
plt.vlines(26.5, -0.1, 0.8, colors='red')
plt.title("Correlations with the Target 'SalePrice'")
X = train_df.drop(columns=['SalePrice'])
y = train_df.SalePrice
less_than_0_corr = train_df.corrwith(train_df['SalePrice'])[train_df.corrwith(train_df['SalePrice']) < 0].index.to_list()
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