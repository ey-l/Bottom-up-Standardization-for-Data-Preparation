import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms
from scipy import stats
from scipy.stats import norm, skew
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
smp = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
plt.figure(figsize=(15, 10))
plt.scatter(train.GrLivArea, train.SalePrice, c='orange', s=90, alpha=0.4)
plt.ylabel('Sales Price', fontsize=15)
plt.xlabel('GrLivArea', fontsize=15)
plt.title('Checking For Outliers', fontsize=15)
plt.grid(alpha=0.5, color='lightslategrey')
sp = plt.gca().spines
sp['top'].set_visible(False)
sp['right'].set_visible(False)
train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)].index, inplace=True)
plt.figure(figsize=(15, 10))
plt.scatter(train.GrLivArea, train.SalePrice, c='orange', s=90, alpha=0.4)
plt.ylabel('Sales Price', fontsize=15)
plt.xlabel('GrLivArea', fontsize=15)
plt.title('Checking For Outliers', fontsize=15)
plt.grid(alpha=0.5, color='lightslategrey')
sp = plt.gca().spines
sp['top'].set_visible(False)
sp['right'].set_visible(False)
(canv, axs) = plt.subplots(2, 2)
canv.set_size_inches(18, 13)
canv.tight_layout(pad=7.0)
title = 'Before'
for rw in range(2):
    plt.sca(axs[rw][0])
    sns.distplot(train['SalePrice'], fit=norm, ax=plt.gca())