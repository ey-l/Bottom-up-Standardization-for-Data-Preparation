import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input3 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
_input4 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
_input5 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
_input1 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
print(_input0.shape)
print(_input0.head(5))
print(_input4.shape)
print(_input4.head(5))
print(_input3.shape)
print(_input3.head(5))
print(_input1.shape)
print(_input1.head(5))
_input0['date'] = pd.to_datetime(_input0['date'], format='%d.%m.%Y')
print(_input0.head(5))
print(_input0.tail(5))
print(_input2.head())
submission = _input2[['ID']]
submission['item_cnt_month'] = 0.188
print(submission.head(5))