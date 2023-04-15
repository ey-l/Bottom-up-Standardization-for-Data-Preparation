
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
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
new_df = df.copy()
for col in df.select_dtypes(include='object').columns:
    new_df = pd.get_dummies(new_df, columns=[col])
df = new_df
df.head()
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
new_df = test_df.copy()
for col in test_df.select_dtypes(include='object').columns:
    new_df = pd.get_dummies(new_df, columns=[col])
test_df = new_df
test_df.head()
idx = 0
for col in df.columns:
    if col not in test_df:
        test_df.insert(idx, col, [0] * len(test_df))
    idx = idx + 1
test_df.head()
test_df = test_df.drop(columns=['SalePrice', 'Id'], axis=1)
y = df['SalePrice']
X = df.drop(columns=['SalePrice', 'Id'], axis=1)
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
train_ratio = 0.9
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=1 - train_ratio, random_state=0)