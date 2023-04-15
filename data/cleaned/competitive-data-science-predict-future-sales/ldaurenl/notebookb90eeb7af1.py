

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
pathToDir = '_data/input/competitive-data-science-predict-future-sales/'
paths = {'pathToDate': pathToDir, 'categories': pathToDir + 'item_categories.csv', 'items': pathToDir + 'items.csv', 'train': pathToDir + 'sales_train.csv', 'submission': pathToDir + 'sample_submission.csv', 'shops': pathToDir + 'shops.csv', 'test': pathToDir + 'test.csv'}
config = {'paths': paths}

def displayInCenter(text):
    outputFormated = '{:*^50}'.format(text)
    print(outputFormated)

def regexFilter(text, regex):
    if not text:
        return False
    return re.search(regex, text)
categoriesDf = pd.read_csv(paths.get('categories'))
itemsDf = pd.read_csv(paths.get('items'))
trainDf = pd.read_csv(paths.get('train'))
submissionDf = pd.read_csv(paths.get('submission'))
shopsDf = pd.read_csv(paths.get('shops'))
testDf = pd.read_csv(paths.get('test'))
initialDatasets = {'categories': categoriesDf, 'items': itemsDf, 'train': trainDf, 'submission': submissionDf, 'shops': shopsDf, 'test': testDf}
displayInCenter('Данные подгрузились')
for (k, v) in initialDatasets.items():
    displayInCenter(k)
    v.info()
displayInCenter('Информация Выведена')
for (k, v) in initialDatasets.items():
    displayInCenter(k)
    print(v.describe())
shopsDf['city'] = shopsDf['shop_name'].apply(lambda x: x.split()[0])
regFilterForCity = lambda city: not regexFilter(city, '^[А-Я]+[А-Я,а-я, ]*$')
strangeCityDf = shopsDf[shopsDf.city.apply(regFilterForCity)].city
displayInCenter('Странные города')
print(strangeCityDf.unique())
displayInCenter('Все Города')
print(shopsDf.city.unique())
trainDf.item_price
sns.distplot(trainDf.item_price, hist=False, kde=True, kde_kws={'linewidth': 3})
sns.distplot(trainDf[trainDf['item_price'] > 25000].item_price, hist=True, kde=True, kde_kws={'linewidth': 3})
sns.distplot(trainDf.item_cnt_day, hist=False, kde=True)
sns.distplot(trainDf[trainDf['item_cnt_day'] > 250].item_cnt_day, hist=True, kde=True)
regFilterForName = lambda name: not regexFilter(name, '^[А-Я]+[А-Я,а-я, "]*$')
shopsDf[['shop_name', 'shop_id']][shopsDf.shop_name.apply(regFilterForName)]
shopsDf[['shop_name', 'shop_id']][shopsDf.city.str.startswith('!Якутск', na=False) | shopsDf.city.str.startswith('Якутск', na=False)]
displayInCenter('Дубликаты shop_name')
print(shopsDf['shop_name'][0], '==', shopsDf['shop_name'][57])
print(shopsDf['shop_name'][1], '==', shopsDf['shop_name'][58])
print(shopsDf['shop_name'][10], '==', shopsDf['shop_name'][11])
trainDf.loc[trainDf['shop_id'] == 0, 'shop_id'] = 57
trainDf.loc[trainDf['shop_id'] == 1, 'shop_id'] = 58
trainDf.loc[trainDf['shop_id'] == 10, 'shop_id'] = 11
testDf.loc[testDf['shop_id'] == 0, 'shop_id'] = 57
testDf.loc[testDf['shop_id'] == 1, 'shop_id'] = 58
testDf.loc[testDf['shop_id'] == 10, 'shop_id'] = 11
shopsDf.loc[shopsDf['city'] == '!Якутск', 'city'] = 'Якутск'
trainDf = trainDf[(trainDf['item_price'] > 0) & (trainDf['item_price'] < 50000)]
trainDf = trainDf[(trainDf['item_cnt_day'] > 0) & (trainDf['item_cnt_day'] < 1000)]
dfAggregator = pd.merge(trainDf, itemsDf, on='item_id', how='left')
dfAggregator = pd.merge(dfAggregator, categoriesDf, on='item_category_id', how='left')
dfAggregator = pd.merge(dfAggregator, shopsDf, on='shop_id', how='left')
dfAggregator['month'] = 1 + dfAggregator['date_block_num'] % 12
dfAggregator['year'] = 2013 + dfAggregator['date_block_num'] // 12
dfAggregator
sns.heatmap(dfAggregator.corr())
features = ['item_id', 'shop_id', 'month', 'year']
train = dfAggregator[['item_id', 'shop_id', 'month', 'year', 'item_cnt_day', 'date_block_num']].groupby(['item_id', 'shop_id', 'month', 'year', 'date_block_num']).sum().reset_index()
train.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
train
testDf['year'] = 2015
testDf['month'] = 11
testDf
from sklearn.model_selection import train_test_split
(train_X, val_X, train_y, val_y) = train_test_split(train[features], train['item_cnt_month'], test_size=0.2, random_state=0)