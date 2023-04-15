import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
df.head(10)
df['item_cnt_day'] = df.item_cnt_day.abs()
df['item_price'] = df.item_price.abs()
df.describe()
df
df = df.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price': 'mean', 'item_cnt_day': 'sum'}).reset_index()
df
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 8))
dataplot = sns.heatmap(df.corr(), cmap='YlGnBu', annot=True)

df['shop_id'].value_counts()
test_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test_data['date_block_num'] = 34
test_data = test_data[['date_block_num', 'shop_id', 'item_id']]
test_data.head()
item_price = dict(df.groupby('item_id')['item_price'].last().reset_index().values)
test_data['item_price'] = test_data.item_id.map(item_price)
print(test_data)
print(df)
test_data['item_price'] = test_data['item_price'].fillna(test_data['item_price'].median())
print(test_data)
df = df.sample(frac=1)
print(df)
from sklearn.model_selection import train_test_split
X = np.array(df.drop(['item_cnt_day'], axis=1))
Y = np.array(df.iloc[:, 4])
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.25, random_state=43)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
poly_regs = PolynomialFeatures(degree=4)
x_poly = poly_regs.fit_transform(X)
clf = LinearRegression()