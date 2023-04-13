import numpy as np
import pandas as pd
from PIL import Image
from dateutil.parser import parse
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import optim
import torch.nn as nn
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
new_df = _input1.copy()
for col in _input1.select_dtypes(include='object').columns:
    new_df = pd.get_dummies(new_df, columns=[col])
_input1 = new_df
_input1.head()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
new_df = _input0.copy()
for col in _input0.select_dtypes(include='object').columns:
    new_df = pd.get_dummies(new_df, columns=[col])
_input0 = new_df
_input0.head()
idx = 0
for col in _input1.columns:
    if col not in _input0:
        _input0.insert(idx, col, [0] * len(_input0))
    idx = idx + 1
_input0.head()
_input0 = _input0.drop(columns=['SalePrice', 'Id'], axis=1)
y = _input1['SalePrice']
X = _input1.drop(columns=['SalePrice', 'Id'], axis=1)
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
train_ratio = 0.9
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=1 - train_ratio, random_state=0)