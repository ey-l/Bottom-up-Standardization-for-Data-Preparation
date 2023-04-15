import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

transactions = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
subm = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
print(transactions.shape)
print(transactions.head(5))
print(items.shape)
print(items.head(5))
print(item_categories.shape)
print(item_categories.head(5))
print(shops.shape)
print(shops.head(5))
transactions['date'] = pd.to_datetime(transactions['date'], format='%d.%m.%Y')
print(transactions.head(5))
print(transactions.tail(5))
print(test.head())
submission = test[['ID']]
submission['item_cnt_month'] = 0.188
print(submission.head(5))
