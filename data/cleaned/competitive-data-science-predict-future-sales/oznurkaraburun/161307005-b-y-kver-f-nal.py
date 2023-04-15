import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = '_data/input/competitive-data-science-predict-future-sales'
items = pd.read_csv(path + '/items.csv')
item_cats = pd.read_csv(path + '/item_categories.csv')
shops = pd.read_csv(path + '/shops.csv')
sales = pd.read_csv(path + '/sales_train.csv')
test = pd.read_csv(path + '/test.csv')
submission = pd.read_csv(path + '/sample_submission.csv')
print('Data set loaded successfully.')
merged = pd.merge(sales, shops, how='left')

print(merged)
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 6))
merged['shop_id'].hist(bins=84)
data = merged.groupby(by=['date'], sort=True, as_index=False).count()
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 8))
plt.title('Satışlar')
plt.plot(data.date, data.item_price)
plt.xlabel('date', fontsize=12)
plt.ylabel('item_price', fontsize=12)

plt.figure(figsize=(16, 8))
plt.title('Satışlar')
plt.scatter(data.date, data.item_price)
plt.xlabel('date', fontsize=12)
plt.ylabel('item_price', fontsize=12)

merged[merged.shop_id > 50]
merged[merged.item_price > 1000]
merged.item_id.value_counts().sort_index().plot.bar()
y = merged.iloc[:, 5].values
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(merged.iloc[:, 1:-1], y, test_size=0.33, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')