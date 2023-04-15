import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import catboost
from sklearn.metrics import mean_squared_error
import scipy
from scipy import stats
from scipy.stats import norm
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', sep=',')
train
test_ID = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
target = 'SalePrice'
print(train.loc[:, target].isnull().any())