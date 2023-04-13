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
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', sep=',')
_input1
test_ID = _input0['Id']
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
target = 'SalePrice'
print(_input1.loc[:, target].isnull().any())