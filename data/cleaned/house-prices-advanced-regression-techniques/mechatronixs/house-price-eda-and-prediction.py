import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
train_dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_dataset.head()

def check_df(dataframe, head=5):
    print('#### Shape #### ')
    print(dataframe.shape)
    print('### Types ###')
    print(dataframe.dtypes)
    print('### Head ###')
    print(dataframe.head(head))
    print('### Tail ###')
    print(dataframe.tail(head))
    print('### NA ###')
    print(dataframe.isnull().sum())
    print('### Quantiles ###')
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)
check_df(train_dataset)

train_dataset.hist(bins=50, figsize=(16, 16))


def grap_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe : dataframe
        değişken isimleri alınmak istenen dataframe'dir.

    cat_th : int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri

    car_th : int,float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
        cat_cols : list
            Kategorik değişken listesi
        num_cols : list
            Numerik değişken listesi
        cat_but_car : list
            Kategorik görünümlü kardinal değişken listesi

    Notes
    -------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_car değişkeni cat_cols'un içerisindedir.
        Return olan 3 liste toplamı, toplam değişken sayısına eşittir.
    """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ['category', 'object', 'bool']]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ['int64', 'float64']]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ['category', 'object']]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ['int64', 'float64']]
    num_cols = [col for col in num_cols if col not in cat_cols]
    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car : {len(cat_but_car)}')
    print(f'num_but_cat : {len(num_but_cat)}')
    return (cat_cols, num_cols, cat_but_car)
(cat_cols, num_cols, cat_but_car) = grap_col_names(train_dataset)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(), 'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('##########################################')
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)

for col in cat_cols:
    cat_summary(train_dataset, col, plot=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)

for col in num_cols:
    num_summary(train_dataset, col, plot=True)
(fig, ax) = plt.subplots(figsize=(10, 6))
ax.grid()
ax.scatter(train_dataset['GrLivArea'], train_dataset['SalePrice'], c='#3f72af', zorder=3, alpha=0.9)
ax.axvline(4500, c='#112d4e', ls='--', zorder=2)
ax.set_xlabel('Ground living area (sq. ft)', labelpad=10)
ax.set_ylabel('Sale price ($)', labelpad=10)

sns.boxplot(train_dataset.GrLivArea)

numerical_df = train_dataset.select_dtypes(exclude=['object'])
numerical_df = numerical_df.drop(['Id'], axis=1)
for column in numerical_df:
    plt.figure(figsize=(16, 4))
    sns.set_theme(style='whitegrid')
    sns.boxplot(numerical_df[column])
(f, ax) = plt.subplots(figsize=(16, 16))
sns.distplot(train_dataset.get('SalePrice'), kde=False)

corrmat = train_dataset.corr()
(f, ax) = plt.subplots(figsize=(16, 16))
sns.heatmap(corrmat, vmax=0.8, square=True)

plt.figure(figsize=(16, 16))
columns = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
correlation_matrix = np.corrcoef(train_dataset[columns].values.T)
sns.set(font_scale=1.25)
heat_map = sns.heatmap(correlation_matrix, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=columns.values, xticklabels=columns.values)

train_dataset = train_dataset[train_dataset.GrLivArea < 4500]
total = test_dataset.isna().sum().sort_values(ascending=False)
missing_data = pd.concat([total], axis=1, keys=['Total'])
missing_data.head(45)
total = total[total > 0]
(fig, ax) = plt.subplots(figsize=(10, 6))
ax.grid()
ax.bar(total.index, total.values, zorder=2, color='#3f72af')
ax.set_ylabel('No. of missing values', labelpad=10)
ax.set_xlim(-0.6, len(total) - 0.4)
ax.xaxis.set_tick_params(rotation=90)

train_dataset = train_dataset.drop(missing_data[missing_data.Total > 0].index, axis=1)
test_dataset = test_dataset.dropna(axis=1)
test_dataset = test_dataset.drop(['Electrical'], axis=1)
full_dataset = pd.concat([train_dataset, test_dataset])
full_dataset = pd.get_dummies(full_dataset)
x = full_dataset.iloc[train_dataset.index]
x_test = full_dataset.iloc[test_dataset.index]
x = x.drop(['SalePrice'], axis=1)
y = train_dataset.SalePrice
from sklearn.model_selection import train_test_split
(x_train, x_val, y_train, y_val) = train_test_split(x, y, train_size=0.8, random_state=42)
x.isna().sum().sort_values(ascending=False)
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
regressor = LinearRegression()