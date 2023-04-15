import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
train_dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_dataset.head()
test_dataset.tail()
train_dataset.fillna(0, inplace=True)
test_dataset.fillna(0, inplace=True)
train_dataset.isnull().sum()
sns.displot(train_dataset['SalePrice'])
copy_train_dataset = train_dataset.copy()

def non_numeric_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit = {}

        def convert_to_int(key):
            return text_digit[key]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_content = df[column].values.tolist()
            unique_elements = set(column_content)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit:
                    text_digit[unique] = x
                    x = x + 1
            df[column] = list(map(convert_to_int, df[column]))
    return df
non_numeric_train = non_numeric_data(copy_train_dataset)
new_test_dataset = non_numeric_data(test_dataset)
new_test_dataset.head()
non_numeric_train.head()
plt.subplots(figsize=(19, 4))
sns.barplot(x=non_numeric_train['YrSold'], y=non_numeric_train['SalePrice'])
plt.xticks(rotation=90)

plt.subplots(figsize=(19, 4))
sns.barplot(x=non_numeric_train['SaleType'], y=non_numeric_train['SalePrice'])
plt.xticks(rotation=90)

plt.subplots(figsize=(19, 4))
sns.barplot(x=train_dataset['Neighborhood'], y=non_numeric_train['SalePrice'])
plt.xticks(rotation=90)

plt.subplots(figsize=(19, 4))
sns.barplot(x=non_numeric_train['OverallQual'], y=non_numeric_train['SalePrice'])
plt.xticks(rotation=90)

numeric_column_names = non_numeric_train.columns.tolist()
correlation = []
for item in numeric_column_names:
    correlation.append(non_numeric_train[item].corr(non_numeric_train['SalePrice']))
correlation_list_df = pd.DataFrame({'column': numeric_column_names, 'correlation': correlation})
correlation_list_df = correlation_list_df.sort_values(by='correlation', ascending=False)
print(correlation_list_df)
plt.subplots(figsize=(19, 4))
sns.barplot(x=correlation_list_df['column'], y=correlation_list_df['correlation'])
plt.xticks(rotation=90)
plt.ylabel('Correlation', fontsize=13)
plt.xlabel('Columns', fontsize=13)
plt.title('Correlation of numeric columns with SalePrice')

non_numeric_train.OverallQual.unique()
quality_pivot = non_numeric_train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
quality_pivot
quality_pivot.plot(kind='bar', color='green')
plt.xlabel('Overall Quality')
plt.ylabel('Median')
plt.xticks(rotation=0)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import svm, preprocessing
from sklearn.metrics import mean_squared_error
X = np.array(non_numeric_train.drop(['SalePrice'], axis=1))
y = np.array(non_numeric_train['SalePrice'])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=42, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
clf = LinearRegression(fit_intercept=False, normalize=False, n_jobs=-1)