import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import datetime
from sklearn.preprocessing import LabelEncoder
from itertools import product
from xgboost import XGBRegressor
from xgboost import plot_importance
route = '_data/input/competitive-data-science-predict-future-sales/'
categorias = pd.read_csv(route + 'item_categories.csv')
items = pd.read_csv(route + 'items.csv')
ventas = pd.read_csv(route + 'sales_train.csv')
tiendas = pd.read_csv(route + 'shops.csv')
test = pd.read_csv(route + 'test.csv')
datos = {'categorias': categorias, 'items': items, 'ventas': ventas, 'tiendas': tiendas, 'test': test}
for (nombre, dato) in datos.items():
    print('Informacion sobre: ' + nombre)
    dato.info()
    print('_' * 50)
    print(' ')
    dato.isnull().sum()
    print('_' * 50)
    print(' ')
ventas['date'] = ventas.date.apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))
ventas['month'] = ventas['date'].dt.month
ventas['days'] = ventas['date'].dt.day
ventas.describe().apply(lambda s: s.apply('{0:.5f}'.format))
neg_price = ventas.loc[ventas['item_price'] < 0]
mean_price_neg = ventas.loc[ventas['shop_id'].isin(neg_price['shop_id']) & ventas['item_id'].isin(neg_price['item_id']), 'item_price'].mean()
ventas.loc[ventas['item_price'] < 0, 'item_price'] = mean_price_neg
neg_items_cnt = ventas.loc[ventas['item_cnt_day'] < 0]
neg_items_cnt.shop_id.value_counts()
ventas = ventas.loc[ventas['item_cnt_day'] >= 0]
(fig, ax) = plt.subplots(2, 1, figsize=(12, 8))
sns.boxplot(ventas['item_price'], ax=ax[0])
sns.boxplot(ventas['item_cnt_day'], ax=ax[1])
ventas.loc[ventas['item_price'] > 50000]
ventas.loc[ventas['item_cnt_day'] > 500]
ventas = ventas.loc[ventas['item_price'] <= 50000]
ventas = ventas.loc[ventas['item_cnt_day'] <= 500]
print('Conjunto de datos de ventas corregidos')
ventas.describe().apply(lambda s: s.apply('{0:.5f}'.format))
ventas_tiempo = ventas.copy()
ventas_tiempo = ventas_tiempo.set_index('date')
total_items_por_mes = ventas_tiempo.item_cnt_day.resample('M').sum()
(fig, ax) = plt.subplots(figsize=(12, 8))
sns.lineplot(data=total_items_por_mes, ax=ax)
ventas_tienda = ventas.groupby(['shop_id'])['item_cnt_day'].sum()
ventas_tienda.sort_values(ascending=False, inplace=True)
ventas_tienda = ventas_tienda[0:10].reset_index()
sns.barplot(y='item_cnt_day', x='shop_id', data=ventas_tienda, order=ventas_tienda.sort_values('item_cnt_day', ascending=False)['shop_id'])
ventas_tienda_item = ventas.groupby(['shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
for tienda in ventas_tienda_item.shop_id.unique():
    agg_ventas_tienda = ventas_tienda_item.loc[ventas_tienda_item['shop_id'] == tienda]
    item_tienda_sort = agg_ventas_tienda.sort_values(['item_cnt_day'], ascending=False)
    print(item_tienda_sort.max())
items.iloc[22167].item_name
ventas['revenue'] = ventas['item_cnt_day'] * ventas['item_price']
ventas_tienda = ventas.groupby(['shop_id'])['revenue'].sum()
ventas_tienda.sort_values(ascending=False, inplace=True)
ventas_tienda = ventas_tienda[0:10].reset_index()
sns.barplot(y='revenue', x='shop_id', data=ventas_tienda, order=ventas_tienda.sort_values('revenue', ascending=False)['shop_id'])
nov_ventas = ventas.loc[ventas['month'] == 11]
group_nov_ventas = nov_ventas.groupby(['shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
top20_nov_items = group_nov_ventas.sort_values(['item_cnt_day'], ascending=False)[0:20]
sns.barplot(y='item_cnt_day', x='shop_id', data=top20_nov_items, order=top20_nov_items.sort_values('item_cnt_day', ascending=False)['shop_id'], hue='item_id')
items.iloc[20949].item_name
list_tiendas = tiendas.shop_name.str.split(' ')
tiendas['city'] = [tienda[0] for tienda in list_tiendas]
tiendas['shop_type'] = [tienda[1] for tienda in list_tiendas]
tiendas.loc[tiendas['city'] == '!Якутск', 'city'] = 'Якутск'
tiendas.loc[tiendas['city'] == 'Интернет-магазин', 'city'] = 'NA'
tiendas['city_id'] = LabelEncoder().fit_transform(tiendas['city'])
tiendas.loc[tiendas['shop_type'] == 'Орджоникидзе,', 'shop_type'] = 'доме'
tiendas.loc[tiendas['shop_type'] == '(Плехановская,', 'shop_type'] = 'доме'
tiendas.loc[tiendas['shop_type'] == '"Распродажа"', 'shop_type'] = 'МТРЦ'
tiendas.loc[tiendas['shop_type'] == 'Посад', 'shop_type'] = 'Интернет-магазин'
tiendas.loc[tiendas['shop_type'] == 'ЧС', 'shop_type'] = 'Интернет-магазин'
tiendas['shop_type_id'] = LabelEncoder().fit_transform(tiendas['shop_type'])
tiendas
ventas.loc[ventas.shop_id == 57, 'shop_id'] = 0
test.loc[test.shop_id == 57, 'shop_id'] = 0
ventas.loc[ventas.shop_id == 58, 'shop_id'] = 1
test.loc[test.shop_id == 58, 'shop_id'] = 1
ventas.loc[ventas.shop_id == 40, 'shop_id'] = 39
test.loc[test.shop_id == 40, 'shop_id'] = 39
ventas.loc[ventas.shop_id == 11, 'shop_id'] = 10
test.loc[test.shop_id == 11, 'shop_id'] = 10
tiendas.drop([57, 58, 40, 11])
list_categorias = categorias.item_category_name.str.split(' - ')
categorias['cat'] = [cat[0] for cat in list_categorias]
subcat = []
for cat in list_categorias:
    if len(cat) > 1:
        subcat.append(cat[1])
    else:
        subcat.append('NA')
categorias['subcat'] = subcat
categorias['cat_id'] = LabelEncoder().fit_transform(categorias['cat'])
categorias['subcat_id'] = LabelEncoder().fit_transform(categorias['subcat'])
categorias.drop('item_category_name', axis=1)
tiendas_ventas = set(ventas.shop_id.unique())
tiendas_test = set(test.shop_id.unique())
print('Todas los tiendas del conjunto de prueba estan en los datos de entrenamiento?')
print(tiendas_test.issubset(tiendas_ventas))
items_ventas = set(ventas.item_id.unique())
items_test = set(test.item_id.unique())
print('Todos los items del conjunto de prueba estan en los datos de entrenamiento?')
print(items_test.issubset(items_ventas))
print('Items que estan en el base de datos de prueba pero no los datos de entrenamiento')
print('Numero de items faltantes: ' + str(len(items_test - items_ventas)))
total_ventas_mes = ventas.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
total_ventas_mes = total_ventas_mes.fillna(0.0)
total_ventas_mes
formato_ventas = []
for date_block in ventas['date_block_num'].unique():
    ventas_mes = ventas.loc[ventas['date_block_num'] == date_block]
    tiendas_unicas = ventas.loc[ventas['date_block_num'] == date_block, 'shop_id'].unique()
    items_unicos = ventas.loc[ventas['date_block_num'] == date_block, 'item_id'].unique()
    formato_ventas.append(np.array(list(product([date_block], tiendas_unicas, items_unicos)), dtype='int16'))
formato_ventas = pd.DataFrame(np.vstack(formato_ventas), columns=['date_block_num', 'shop_id', 'item_id']).reset_index()
formato_ventas.sort_values('date_block_num', inplace=True)
total_ventas = pd.merge(formato_ventas, total_ventas_mes, on=['date_block_num', 'shop_id', 'item_id'], how='left')
total_ventas = total_ventas.fillna(0)
total_ventas = total_ventas.drop('index', axis=1)
total_ventas['item_cnt_day'] = total_ventas['item_cnt_day'].astype('float16')
del formato_ventas
del total_ventas_mes
del ventas_tiempo
del ventas_tienda_item
del ventas_mes
del top20_nov_items
del group_nov_ventas
del item_tienda_sort
del neg_items_cnt
del total_items_por_mes
del agg_ventas_tienda
del fig, ax
gc.collect()
test.drop(['ID'], axis=1, inplace=True)
test['date_block_num'] = 34
total_ventas = pd.concat([total_ventas, test], ignore_index=True, sort=False, keys=['date_block_num', 'shop_id', 'item_id'])
total_ventas.fillna(0, inplace=True)
total_ventas['date_block_num'] = total_ventas['date_block_num'].astype('int16')
total_ventas['shop_id'] = total_ventas['shop_id'].astype('int16')
total_ventas['item_id'] = total_ventas['item_id'].astype('int16')
items_cat = pd.merge(items, categorias, on='item_category_id', how='left')
items_cat.drop(['item_name', 'item_category_name', 'item_category_id', 'cat', 'subcat'], axis=1, inplace=True)
total_ventas = pd.merge(total_ventas, items_cat, on='item_id', how='left')
total_ventas['cat_id'] = total_ventas['cat_id'].astype('int16')
total_ventas['subcat_id'] = total_ventas['subcat_id'].astype('int16')
del items_cat
tiendas_var = tiendas.drop(['shop_name', 'shop_type', 'city'], axis=1)
total_ventas = pd.merge(total_ventas, tiendas_var, on='shop_id', how='left')
total_ventas['city_id'] = total_ventas['city_id'].astype('int16')
total_ventas['shop_type_id'] = total_ventas['shop_type_id'].astype('int16')
del tiendas_var
total_ventas['month'] = total_ventas['date_block_num'] % 12 + 1

def lags_vars(target_df, target_cols, lags=[1, 6, 12]):
    std_cols = ['date_block_num', 'shop_id', 'item_id']
    for col in target_cols:
        std_cols.append(col)
    original = target_df.loc[:, std_cols]
    for lag in lags:
        for col in target_cols:
            shift = original.copy()
            colname = 'lag_' + str(lag) + '_' + col
            shift.rename(columns={col: colname}, inplace=True)
        shift.loc[:, 'date_block_num'] = shift.loc[:, 'date_block_num'] + lag
        target_df = pd.merge(target_df, shift, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return target_df
total_ventas = lags_vars(total_ventas, ['item_cnt_day']).fillna(0)

def aggregate_data(data_df, target_df, cols, fun, name, target='item_cnt_day'):
    agg_values = data_df.groupby(cols)[target].apply(lambda x: fun(x)).reset_index()
    agg_values = agg_values.rename({target: name}, axis=1)
    target_df = pd.merge(target_df, agg_values, on=cols, how='left').fillna(0)
    target_df[name] = target_df[name].astype('float16')
    return target_df
total_ventas = aggregate_data(data_df=ventas, target_df=total_ventas, cols=['date_block_num', 'shop_id'], fun=np.mean, name='mean_shop_block')
total_ventas = aggregate_data(data_df=ventas, target_df=total_ventas, cols=['date_block_num', 'item_id'], fun=np.mean, name='mean_item_block')
items_cat = pd.merge(items, categorias, on='item_category_id', how='left')
items_cat.drop(['item_name', 'item_category_name', 'item_category_id', 'cat', 'subcat'], axis=1, inplace=True)
ventas_cat = pd.merge(ventas, items_cat, on='item_id', how='left')
total_ventas = aggregate_data(data_df=ventas_cat, target_df=total_ventas, cols=['date_block_num', 'cat_id'], fun=np.mean, name='mean_cat_block')
total_ventas = aggregate_data(data_df=ventas, target_df=total_ventas, target='item_price', cols=['date_block_num', 'item_id'], fun=np.mean, name='mean_item_price')
total_ventas
total_ventas = lags_vars(total_ventas, ['mean_shop_block']).fillna(0)
total_ventas = lags_vars(total_ventas, ['mean_item_block']).fillna(0)
total_ventas = lags_vars(total_ventas, ['mean_cat_block']).fillna(0)
total_ventas = lags_vars(total_ventas, ['mean_item_price']).fillna(0)
total_ventas.columns
total_ventas.to_pickle('datos_final.pkl')
del ventas
del items
del categorias
del tiendas
del dato
del total_ventas
del ventas_cat
del aggregate_data
del neg_price

final_data = pd.read_pickle('datos_final.pkl')
train_set = final_data.loc[final_data['date_block_num'] < 33]
x_train = train_set.drop('item_cnt_day', axis=1)
y_train = train_set['item_cnt_day']
val_set = final_data.loc[final_data['date_block_num'] == 33]
x_val = val_set.drop('item_cnt_day', axis=1)
y_val = val_set['item_cnt_day']
test_set = final_data.loc[final_data['date_block_num'] == 34]
x_test = test_set.drop('item_cnt_day', axis=1)
y_test = test_set['item_cnt_day']
del train_set
del val_set
del test_set
model = XGBRegressor(max_depth=10, n_estimators=1000, min_child_weight=100, colsample_bytree=0.8, subsample=1, eta=0.2)