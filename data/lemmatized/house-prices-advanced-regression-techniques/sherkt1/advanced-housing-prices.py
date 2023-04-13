import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import train_test_split
import xgboost as xgb
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')

def checkcategories(train, test, fthreshold):
    trt = pd.concat((_input1, _input0), axis=0)
    trtshape = trt.shape
    alt = np.zeros(trtshape[0])
    tru = _input1.unique()
    teu = _input0.unique()
    for values in teu:
        if (values in tru) == False:
            trt[(trt == values) == True] = 'zzztempzzz'
    trt[trt.isnull() == True] = 'zzztempzzz'
    tdict = trt.value_counts().to_dict()
    for values in tdict:
        if tdict[values] < fthreshold:
            trt[(trt == values) == True] = 'zzztempzzz'
    dummies = pd.get_dummies(trt)
    _input1 = dummies.iloc[0:_input1.shape[0], :]
    _input0 = dummies.iloc[_input1.shape[0]:_input1.shape[0] + _input0.shape[0] + 1, :]
    return (_input1, _input0)
(traintemp, testtemp) = checkcategories(_input1.loc[:, 'Fence'], _input0.loc[:, 'Fence'], 100)
catcolumns = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 39, 40, 41, 42, 47, 48, 49, 50, 52, 53, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 72, 73, 74, 77, 78, 79]
numcolumns = [3, 4, 17, 18, 19, 20, 34, 36, 37, 38, 43, 44, 45, 46, 51, 54, 62, 66, 67, 68, 69, 70, 71, 75, 76]
for (count, values) in enumerate(catcolumns):
    (traintemp, testtemp) = checkcategories(_input1.iloc[:, values], _input0.iloc[:, values], 100)
    if count == 0:
        traincat = traintemp.to_numpy()
        testcat = testtemp.to_numpy()
    else:
        traincat = np.concatenate((traincat, traintemp.to_numpy()), axis=1)
        testcat = np.concatenate((testcat, testtemp.to_numpy()), axis=1)
for (count, values) in enumerate(numcolumns):
    traintemp = _input1.iloc[:, values]
    testtemp = _input0.iloc[:, values]
    mx = max(traintemp.max(), testtemp.max())
    mi = min(traintemp.min(), testtemp.min())
    traintemp = (traintemp - mi) / (mx - mi)
    testtemp = (testtemp - mi) / (mx - mi)
    a = traintemp.to_numpy()
    b = testtemp.to_numpy()
    temp = np.concatenate((a.reshape(-1, 1), b.reshape(-1, 1)))
    s = skew(temp)
    if s > 0.75:
        temp = np.log1p(temp)
    traintemp = temp[0:traintemp.shape[0]]
    testtemp = temp[traintemp.shape[0]:traintemp.shape[0] + testtemp.shape[0]]
    if count == 0:
        trainnum = traintemp
        testnum = testtemp
    else:
        trainnum = np.concatenate([trainnum, traintemp], axis=1)
        testnum = np.concatenate([testnum, testtemp], axis=1)
trainall = np.concatenate((traincat, trainnum), axis=1)
testall = np.concatenate((testcat, testnum), axis=1)
trainall[np.isnan(trainall) == True] = 0.5
testall[np.isnan(testall) == True] = 0.5
(X_train, X_test, y_train, y_test) = train_test_split(trainall, _input1.iloc[:, 80].to_numpy(), test_size=0.2, random_state=42)
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.7, learning_rate=0.1, max_depth=3, reg_lambda=0.8, n_estimators=500, evaluation_metric='rmsle')