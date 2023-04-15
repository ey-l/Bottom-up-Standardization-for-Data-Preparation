import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
val = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')), 'item_id', 'shop_id']).sum().reset_index()
df = df[['date', 'item_id', 'shop_id', 'item_cnt_day']]
df['item_cnt_day'].clip(0.0, 20.0, inplace=True)
df = df.pivot_table(index=['item_id', 'shop_id'], columns='date', values='item_cnt_day', fill_value=0).reset_index()
df = pd.merge(val, df, on=['item_id', 'shop_id'], how='left').fillna(0)
main = pd.DataFrame()
previous_month = pd.DataFrame(df.iloc[:, -1].values, columns=['item_cnt_month'])
main['previous_month'] = df.iloc[:, -1].values

previous_month.head()
pred = df.iloc[:, -3:].mean(axis=1)
main['threemonthmean'] = pred
threemonthmean = pd.DataFrame(pred.values, columns=['item_cnt_month'])

threemonthmean.head()
pred = df.iloc[:, -3:].mean(axis=1)
main['twomonthmean'] = pred
twomonthmean = pd.DataFrame(pred.values, columns=['item_cnt_month'])

twomonthmean.head()
pred = df.iloc[:, -5:].mean(axis=1)
main['fivemonthmean'] = pred
fivemonthmean = pd.DataFrame(pred.values, columns=['item_cnt_month'])

fivemonthmean.head()
df.head()
pred = df.iloc[:, 4:].mean(axis=1)
main['fullmean'] = pred
fullmean = pd.DataFrame(pred.values, columns=['item_cnt_month'])

fullmean.head()
pred = df.iloc[:, -2:].median(axis=1)
main['median'] = pred
median = pd.DataFrame(pred.values, columns=['item_cnt_month'])

median.head()
essemble_df = main[['threemonthmean', 'twomonthmean', 'fivemonthmean', 'median']]
pred = essemble_df.mean(axis=1)
main['mean_essemble'] = pred
mean_essemble = pd.DataFrame(pred.values, columns=['item_cnt_month'])

mean_essemble.head()
main_melt = pd.melt(main)
sns.boxplot(data=main_melt, x='value', y='variable')
plt.title('Distribution of Submissions')
plt.xlabel('Predict Items Sold')
plt.xlim(-0.5, 2)
