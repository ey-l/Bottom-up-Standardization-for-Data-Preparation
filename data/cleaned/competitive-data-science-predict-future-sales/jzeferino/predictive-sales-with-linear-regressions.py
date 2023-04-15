import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import re
path = '_data/input/competitive-data-science-predict-future-sales/'
train = pd.read_csv(path + 'sales_train.csv')
test = pd.read_csv(path + 'test.csv')
sample_submission = pd.read_csv(path + 'sample_submission.csv')
items = pd.read_csv(path + 'items.csv')
item_categories = pd.read_csv(path + 'item_categories.csv')
shops = pd.read_csv(path + 'shops.csv')
print('************** SALES_TRAIN **************')
print(train.describe())
print(train.head(6))
print('************** TEST **************')
print(test.head(3))
print(test.describe())
print('************** OTHERS **************')
print(sample_submission.head(3))
print(items.head(1))
print(item_categories.head(1))
print(shops.head(1))
print('Check for Nulls:')
print(train.isnull().sum())
print(test.isnull().sum())
dif1_a = list(set(train['shop_id']) - set(test['shop_id']))
print('Dif shop_id - TRAIN NOT TEST: ', dif1_a)
dif1_b = list(set(test['shop_id']) - set(train['shop_id']))
print('Dif shop_id - TEST NOT TRAIN: ', dif1_b)
dif2_a = list(set(train['item_id']) - set(test['item_id']))
print('Amount Dif item_id - TRAIN NOT TEST: ', len(dif2_a))
dif2_b = list(set(test['item_id']) - set(train['item_id']))
print('Amount Dif item_id - TEST NOT TRAIN: ', len(dif2_b))
print('There are ', len(dif2_a), ' items not sold and ', len(dif2_b), ' new.')
train_grouped_d_s = train.groupby(['date_block_num', 'shop_id'], as_index=False)['item_cnt_day'].sum()
train_grouped_d_i = train.groupby(['date_block_num', 'item_id'], as_index=False)['item_cnt_day'].sum()
train_grouped_d_s_i = train.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)['item_cnt_day'].sum()
train_grouped_d = train.groupby(['date_block_num'], as_index=False)['item_cnt_day'].sum()
print('************** GROUPED BY DATE AND SHOP_ID **************')
print(train_grouped_d_s.describe())
print(train_grouped_d_s.head(3))
print('************** GROUPED BY DATE AND ITEM_ID **************')
print(train_grouped_d_i.describe())
print(train_grouped_d_i.head(3))
print('************** GROUPED BY DATE AND SHOP_ID AND ITEM_ID **************')
print(train_grouped_d_s_i.describe())
print(train_grouped_d_s_i.head(3))
print('************** GROUPED BY DATE **************')
print(train_grouped_d.describe())
print(train_grouped_d.head(3))
train_grouped_s = train.groupby(['shop_id'], as_index=False)['item_cnt_day'].sum()
train_grouped_i = train.groupby(['item_id'], as_index=False)['item_cnt_day'].sum()
train_grouped_s_i = train.groupby(['shop_id', 'item_id'], as_index=False)['item_cnt_day'].sum()
print('************** GROUPED BY SHOP_ID **************')
print(train_grouped_s.describe())
print(train_grouped_s.head(3))
print('************** GROUPED BY ITEM_ID **************')
print(train_grouped_i.describe())
print(train_grouped_i.head(3))
print('************** GROUPED BY SHOP_ID AND ITEM_ID **************')
print(train_grouped_s_i.describe())
print(train_grouped_s_i.head(3))
fig1 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d)
fig1a = fig1.fig
fig1a.suptitle('Total solds Time Series', fontsize=12)
fig2 = sns.relplot(x='item_id', y='item_cnt_day', data=train_grouped_i)
fig2a = fig2.fig
fig2a.suptitle('Items sold', fontsize=12)
fig3 = sns.relplot(x='shop_id', y='item_cnt_day', data=train_grouped_s)
fig3a = fig3.fig
fig3a.suptitle('Solds per Shop', fontsize=12)
print('*** Item outlier ***')
for i in range(1, len(train_grouped_i)):
    if train_grouped_i.iloc[i, 1] >= 25000:
        print(train_grouped_i.iloc[i, 0], ' -> ', items.iloc[i, 0])
print('*** Biggest shop ***')
for i in range(1, len(train_grouped_s)):
    if train_grouped_s.iloc[i, 1] >= 250000:
        print(train_grouped_s.iloc[i, 0], ' -> ', shops.iloc[i, 0])
print("*** Check if Best Seller item (outlier) is 'legitim' ***")
fig4 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_i.loc[train_grouped_d_i['item_id'] == 20949])
fig4a = fig4.fig
fig4a.suptitle('Solds item 20949', fontsize=12)
print('*** Check Biggest shop follows the typical trend ***')
fig5 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_s.loc[train_grouped_d_s['shop_id'] == 31])
fig5a = fig5.fig
fig5a.suptitle('Solds Shop 31', fontsize=12)
print('*** Check Best Seller item at Biggest shop ***')
fig6 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_s_i.loc[(train_grouped_d_s_i['item_id'] == 20949) & (train_grouped_d_s_i['shop_id'] == 31)])
fig6a = fig6.fig
fig6a.suptitle('Solds of item 20949 at Shop 31', fontsize=12)
print('*** Check other Shops ***')
fig5 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_s.loc[train_grouped_d_s['shop_id'] == 2])
fig5a = fig5.fig
fig5a.suptitle('Solds Shop 2', fontsize=12)
fig6 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_s_i.loc[(train_grouped_d_s_i['item_id'] == 20949) & (train_grouped_d_s_i['shop_id'] == 2)])
fig6a = fig6.fig
fig6a.suptitle('Solds of item 20949 at Shop 2', fontsize=12)
fig5 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_s.loc[train_grouped_d_s['shop_id'] == 12])
fig5a = fig5.fig
fig5a.suptitle('Solds Shop 12', fontsize=12)
fig6 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_s_i.loc[(train_grouped_d_s_i['item_id'] == 20949) & (train_grouped_d_s_i['shop_id'] == 12)])
fig6a = fig6.fig
fig6a.suptitle('Solds of item 20949 at Shop 12', fontsize=12)
print('*** Check other items ***')
print(items.loc[items['item_id'] == 7223])
print(train_grouped_d_i.loc[train_grouped_d_i['item_id'] == 7223, 'item_cnt_day'].sum())
print(items.loc[items['item_id'] == 3731])
print(train_grouped_d_i.loc[train_grouped_d_i['item_id'] == 3731, 'item_cnt_day'].sum())
print(items.loc[items['item_id'] == 3733])
print(train_grouped_d_i.loc[train_grouped_d_i['item_id'] == 3733, 'item_cnt_day'].sum())
fig4 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_i.loc[train_grouped_d_i['item_id'] == 3731])
fig4a = fig4.fig
fig4a.suptitle('Solds item 3731', fontsize=12)
fig6 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_s_i.loc[(train_grouped_d_s_i['item_id'] == 3731) & (train_grouped_d_s_i['shop_id'] == 12)])
fig6a = fig6.fig
fig6a.suptitle('Solds of item 3731 at Shop 12', fontsize=12)
fig6 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_s_i.loc[(train_grouped_d_s_i['item_id'] == 3731) & (train_grouped_d_s_i['shop_id'] == 31)])
fig6a = fig6.fig
fig6a.suptitle('Solds of item 3731 at Shop 31', fontsize=12)
fig4 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_i.loc[train_grouped_d_i['item_id'] == 3733])
fig4a = fig4.fig
fig4a.suptitle('Solds item 3733', fontsize=12)
print('*** Identify type of items atypical ***')
print(10447, ' -> ', items.loc[10447]['item_name'])
print(17717, ' -> ', items.loc[17717]['item_name'])
print(2293, ' -> ', items.loc[2293]['item_name'])
print(3460, ' -> ', items.loc[3460]['item_name'])
print(1555, ' -> ', items.loc[1555]['item_name'])
print(6675, ' -> ', items.loc[6675]['item_name'])
print(20404, ' -> ', items.loc[20404]['item_name'])
print(20405, ' -> ', items.loc[20405]['item_name'])
fig4 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_i.loc[train_grouped_d_i['item_id'] == 10447])
fig4a = fig4.fig
fig4a.suptitle('Solds item 10447', fontsize=12)
fig4 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_i.loc[train_grouped_d_i['item_id'] == 17717])
fig4a = fig4.fig
fig4a.suptitle('Solds item 17717', fontsize=12)
fig4 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_i.loc[train_grouped_d_i['item_id'] == 2293])
fig4a = fig4.fig
fig4a.suptitle('Solds item 2293', fontsize=12)
fig4 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_i.loc[train_grouped_d_i['item_id'] == 3460])
fig4a = fig4.fig
fig4a.suptitle('Solds item 3460', fontsize=12)
fig4 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_i.loc[train_grouped_d_i['item_id'] == 1555])
fig4a = fig4.fig
fig4a.suptitle('Solds item 1555', fontsize=12)
fig4 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_i.loc[train_grouped_d_i['item_id'] == 6675])
fig4a = fig4.fig
fig4a.suptitle('Solds item 6675', fontsize=12)
fig4 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_i.loc[train_grouped_d_i['item_id'] == 20404])
fig4a = fig4.fig
fig4a.suptitle('Solds item 20404', fontsize=12)
fig4 = sns.relplot(x='date_block_num', y='item_cnt_day', data=train_grouped_d_i.loc[train_grouped_d_i['item_id'] == 20405])
fig4a = fig4.fig
fig4a.suptitle('Solds item 20405', fontsize=12)
aty_items = []
print('русские субтитр - Russian subtitles ')
for i in range(0, len(items['item_name'])):
    txt = items.loc[i]['item_name']
    x = re.search('русские субтитры', txt)
    if x:
        aty_items.append(i)
print('русские субтитр - Russian version ')
for i in range(0, len(items['item_name'])):
    txt = items.loc[i]['item_name']
    x = re.search('русская версия', txt)
    if x:
        aty_items.append(i)
print(aty_items)
stg_dsi = train_grouped_d_s.copy()
for i in dif1_a:
    stg_dsi.drop(stg_dsi[stg_dsi['shop_id'] == i].index, inplace=True)
dif10_a = list(set(stg_dsi['shop_id']) - set(test['shop_id']))
print('Dif shop_id A: ', dif10_a)
print('*** All shops ***')
g = sns.FacetGrid(stg_dsi, col='shop_id', col_wrap=4, height=2, xlim=(10, 35), ylim=(0, 5000))
g.map(sns.regplot, 'date_block_num', 'item_cnt_day')
test_s_list = np.unique(test['shop_id'])
print(test_s_list)
shop_func1 = []
for j in range(0, len(test_s_list)):
    train_shopj1 = stg_dsi.loc[stg_dsi['shop_id'] == test_s_list[j]]
    t_data1 = np.array(train_shopj1['date_block_num'])
    y_data1 = np.array(train_shopj1['item_cnt_day'])

    def func(t_data1, a, b):
        return a * t_data1 + b
    InitialParams1 = [0.1, 1000]
    (fitParams1, pcov1) = curve_fit(func, t_data1, y_data1, p0=InitialParams1, method='dogbox')
    shop_func1.append(fitParams1)
print('November 15th, 2015 , #1 ->')
y_month1 = []
for j in range(0, len(test_s_list)):
    sells1 = shop_func1[j][0] * 34 + shop_func1[j][1]
    print(sells1)
    y_month1.append(sells1)
    if sells1 <= 0:
        print(j)
shop_func2 = []
for j in range(0, len(test_s_list)):
    train_shopj2 = stg_dsi.loc[(stg_dsi['shop_id'] == test_s_list[j]) & (stg_dsi['date_block_num'] >= 31)]
    t_data2 = np.array(train_shopj2['date_block_num'])
    y_data2 = np.array(train_shopj2['item_cnt_day'])

    def func(t_data2, a, b):
        return a * t_data2 + b
    InitialParams2 = [0.1, 1000]
    (fitParams2, pcov2) = curve_fit(func, t_data2, y_data2, p0=InitialParams2, method='dogbox')
    shop_func2.append(fitParams2)
print('November 15th, 2015, #2 ->')
y_month2 = []
for j in range(0, len(test_s_list)):
    sells2 = shop_func2[j][0] * 34 + shop_func2[j][1]
    print(sells2)
    y_month2.append(sells2)
    if sells2 <= 0:
        print(j)
y_month = []
for j in range(0, len(y_month1)):
    y_month.append((y_month1[j] + y_month2[j]) / 2)
print(y_month)
perc_item_s = train_grouped_d_s_i.copy()
print('perc_item_s - Before - All months')
print('Total = ', perc_item_s['item_cnt_day'].sum())
print('Shop2 = ', perc_item_s.loc[perc_item_s['shop_id'] == 2, 'item_cnt_day'].sum())
print('Item 20949 at Shop2 = ', perc_item_s.loc[(perc_item_s['item_id'] == 20949) & (perc_item_s['shop_id'] == 2), 'item_cnt_day'].sum())
perc_item_s = perc_item_s.loc[(perc_item_s['date_block_num'] == 22) | (perc_item_s['date_block_num'] >= 31)]
for i in range(len(aty_items)):
    perc_item_s.loc[(perc_item_s['date_block_num'] == 22) & (perc_item_s['item_id'] == aty_items[i])] = perc_item_s.loc[(perc_item_s['date_block_num'] == 33) & (perc_item_s['item_id'] == aty_items[i])]
perc_item_s.loc[(perc_item_s['date_block_num'] == 22) & (perc_item_s['item_id'] == 20949)] = perc_item_s.loc[(perc_item_s['date_block_num'] == 33) & (perc_item_s['item_id'] == 20949)]
perc_item_s.loc[(perc_item_s['date_block_num'] == 22) & (perc_item_s['item_id'] == 6675)] = perc_item_s.loc[(perc_item_s['date_block_num'] == 33) & (perc_item_s['item_id'] == 6675)]
print('perc_item_s - After - Reduced months')
print('Total = ', perc_item_s['item_cnt_day'].sum())
print('Shop2 = ', perc_item_s.loc[perc_item_s['shop_id'] == 2, 'item_cnt_day'].sum())
print('Item 20949 at Shop2 = ', perc_item_s.loc[(perc_item_s['item_id'] == 20949) & (perc_item_s['shop_id'] == 2), 'item_cnt_day'].sum())
perc_item_s = perc_item_s.groupby(['shop_id', 'item_id'], as_index=False)['item_cnt_day'].sum()
for i in dif1_a:
    perc_item_s.drop(perc_item_s[perc_item_s['shop_id'] == i].index, inplace=True)
print('perc_item_s drop shops - head: ', perc_item_s.head(5))
print('Total sells in these shops  = ', perc_item_s['item_cnt_day'].sum())
print('Amount of items in TEST: ', len(test['item_id'].unique()))
print('Amount of items in perc_item_s: ', len(perc_item_s['item_id'].unique()))
dif3_b = list(set(test['item_id']) - set(perc_item_s['item_id']))
print('Amount Dif item_id - TEST NOT perc_item_s: ', len(dif3_b))
dif4_b = list(set(perc_item_s['item_id']) - set(test['item_id']))
print('Amount Dif item_id - perc_item_s NOT TEST: ', len(dif4_b))
train_s2 = train.copy()
train_s2 = train_s2.loc[(train_s2['date_block_num'] == 22) | (train_s2['date_block_num'] >= 31)]
print(len(train_s2))
print('train_s2 - After - Reduced months')
print('Total = ', train_s2['item_cnt_day'].sum())
print('Shop2 = ', train_s2.loc[train_s2['shop_id'] == 2, 'item_cnt_day'].sum())
print('Item 20949 at Shop2 = ', train_s2.loc[(train_s2['item_id'] == 20949) & (train_s2['shop_id'] == 2), 'item_cnt_day'].sum())
train_s2 = train_s2.groupby(['shop_id'], as_index=False)['item_cnt_day'].sum()
print('train_s2 - head: ', train_s2.head(5))
dif5_a = list(set(perc_item_s['shop_id']) - set(train_s2['shop_id']))
dif5_b = list(set(train_s2['shop_id']) - set(perc_item_s['shop_id']))
print('Amount Dif item_id - perc_item_s NOT train_s2: ', dif5_a)
print('Amount Dif item_id - train_s2 NOT perc_item_s: ', dif5_b)
perc_item_s['sh_sum'] = np.nan
for i in range(len(perc_item_s['shop_id'])):
    aa = train_s2.loc[train_s2['shop_id'] == perc_item_s.iat[i, 0]]['item_cnt_day'].tolist()
    perc_item_s.iat[i, 3] = aa[0]
perc_item_s['percent'] = perc_item_s['item_cnt_day'] / perc_item_s['sh_sum']
print('perc_item_s - head: ', perc_item_s.head(5))
print('perc_item_s - tail: ', perc_item_s.tail(5))
print(perc_item_s.loc[perc_item_s['shop_id'] == 31])
print(perc_item_s.loc[(perc_item_s['shop_id'] == 31) & (perc_item_s['item_id'] == 20949)])
print('shop 31 percentage sum: ', perc_item_s.loc[perc_item_s['shop_id'] == 31]['percent'].sum())
print(perc_item_s.loc[perc_item_s['shop_id'] == 5])
print('shop 5 percentage sum: ', perc_item_s.loc[perc_item_s['shop_id'] == 5]['percent'].sum())
test_res = pd.merge(test, perc_item_s, how='left', on=['shop_id', 'item_id'])
print('Check if percentages total in new test_res set are lower')
print(test_res.loc[test_res['shop_id'] == 31]['percent'].sum())
print(test_res.loc[test_res['shop_id'] == 5]['percent'].sum())
test_res['final_res'] = np.nan
print(test_res)
for i in range(len(test_res['shop_id'])):
    bb = test_res.loc[test_res['shop_id'] == test_res.iat[i, 1]]
print('Sells per shop per item in Nov 2015')
for i in range(0, len(test_res['ID'])):
    posit = np.where(test_s_list == test_res.iat[i, 1])[0][0]
    test_res.iat[i, 6] = y_month[posit] * test_res.iat[i, 5]
print(test_res)
tot_estimate = test_res['final_res'].sum()
print('Total sells estimated = ', tot_estimate)
print('There are several NAs, that correspond to new products, or products that where not in previous months used in estimation')
print(test_res.head(5))
print('Amount of items in TEST: ', len(test['item_id'].unique()))
print('Amount of items in test_res: ', len(test_res['item_id'].unique()))
dif5_b = list(set(test['item_id']) - set(test_res['item_id']))
print('Amount Dif item_id - TEST NOT test_res: ', len(dif5_b))
dif6_b = list(set(test_res['item_id']) - set(test['item_id']))
print('Amount Dif item_id - test_res NOT TEST: ', len(dif6_b))
t_aux = []
for j in range(0, len(test_s_list)):
    t_aux0 = test_res.loc[test_res['shop_id'] == test_s_list[j]]
    t_aux.append([test_s_list[j], t_aux0['final_res'].mean() * 0.2])
print(t_aux)
for i in range(0, len(test_res['final_res'])):
    if np.isnan(test_res.iat[i, 6]) == True:
        posaux = np.where(test_s_list == test_res.iat[i, 1])[0][0]
        test_res.iloc[i, 6] = t_aux[posaux][1]
print('test_res final', test_res)
print('Final total sells estimated = ', test_res['final_res'].sum())
test_res_grouped = test_res.groupby(['shop_id'], as_index=False)['final_res'].sum()
test_res_grouped['date_block_num'] = 34
test_res_grouped.rename(columns={'final_res': 'item_cnt_day'}, inplace=True)
train_test_res = pd.concat([stg_dsi, test_res_grouped])
print(train_test_res.head())
print(train_test_res.tail())
print('*** All shops ***')
g = sns.FacetGrid(train_test_res, col='shop_id', col_wrap=4, height=2, xlim=(25, 36), ylim=(100, 7500))
g.map(sns.regplot, 'date_block_num', 'item_cnt_day')
fig7 = sns.relplot(x='ID', y='final_res', data=test_res)
fig7a = fig7.fig
fig7a.suptitle('Solds at 11/2015', fontsize=12)
fig9 = sns.relplot(x='ID', y='final_res', data=test_res.loc[test_res['shop_id'] == 31])
fig9a = fig9.fig
fig9a.suptitle('Solds Shop 31 at 11/2015', fontsize=12)
fig10 = sns.relplot(x='ID', y='final_res', data=test_res.loc[(test_res['item_id'] == 20949) & (test_res['shop_id'] == 31)])
fig10a = fig10.fig
fig10a.suptitle('Solds of item 20949 at Shop 31 at 11/2015', fontsize=12)
submit = test_res[['ID', 'final_res']]
submit.columns = ['ID', 'item_cnt_month']
print('Clip into [0,20] range')
submit['item_cnt_month'][submit['item_cnt_month'] > 20] = 20
submit['item_cnt_month'][submit['item_cnt_month'] < 0] = 0
print(submit.head(10))
print(submit.tail(10))
print('Final checks')
print(submit[46359:46362])
print(submit[45:55])
print(submit[460:465])
print('Final total sells estimated, after clip = ', submit['item_cnt_month'].sum())
