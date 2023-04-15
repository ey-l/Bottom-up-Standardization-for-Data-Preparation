import numpy as np
import pandas as pd
import datetime
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sales.head()
sales.date = sales.date.apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))
sales.head()
sales.info()
sales.isna()
sales.isnull().sum().sum()
sales.describe()
sales.loc[sales['item_id'] == 6675]
print(sales['item_cnt_day'].agg(['sum']))
test.head()
items.tail()
items.isnull().sum(axis=0)
items.isnull().sum(axis=1)
item_categories.head()
shops.head()
newSales = pd.merge(sales, items, on=['item_id', 'item_id']).drop(['item_name'], 1)
newSales.head(70)

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype in ['int64', 'int32']]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df
dsize_sales = downcast_dtypes(newSales)
print(dsize_sales.info())
dsize_sales.head()
dsize_sales['revenue'] = dsize_sales['item_cnt_day'] * dsize_sales['item_price']
dsize_sales.head()
rankRevenue = dsize_sales[['shop_id', 'revenue']]
rankRevenue['revenue'] = round(rankRevenue['revenue'])
topShops = rankRevenue.groupby(['shop_id']).sum().sort_values('revenue', ascending=False)
topShops.head()
topShops['revenue'] = round(topShops['revenue'] / 1000000, 2)
topShops.head()
topShops.tail()
rankItems = dsize_sales[['item_id', 'revenue']]
topItems = rankItems.groupby(['item_id'], as_index=False).sum().sort_values('revenue', ascending=False)
topItems['revenue(m)'] = round(topItems['revenue'] / 1000000)
topItems['index'] = range(1, len(topItems) + 1)
topItems.head()
topItems.tail()
topItems['item_id'] = np.where(topItems['index'] <= 20, topItems['item_id'], 'Other')
newTop = topItems.groupby('item_id', as_index=False).sum().sort_values('revenue', ascending=False).reset_index()
newTop = newTop[['item_id', 'revenue(m)']]
newTop.head()
print(newTop['revenue(m)'].agg(['sum']))
newTop['percent'] = (newTop['revenue(m)'] / 2631 * 100).astype(float).round(2)
import plotly.graph_objects as go
fig = go.Figure(data=[go.Table(header=dict(values=list(newTop.columns), fill_color='green', font=dict(color='white', size=12), align='left'), cells=dict(values=[newTop.item_id, newTop['revenue(m)'], newTop.percent], line_color='darkslategray', fill_color='light green', align='left'))])
fig.update_layout(title='Top-selling items by resevene(millions ruble)from 2013 January to 2015 October', yaxis_zeroline=False, xaxis_zeroline=False)
fig.show()
newTop.tail()
top20Items = topItems.nlargest(20, ['revenue(m)'])
import plotly.express as px
fig = px.pie(top20Items, values='revenue(m)', names='item_id', title='The top 20 items by revenue from 2013 January to 2015 October (Ruple-millions)')
fig.show()
rankSoldPieces = dsize_sales[['item_id', 'item_cnt_day']]
topSoldPCs = rankSoldPieces.groupby(by=['item_id'], as_index=False).sum().sort_values('item_cnt_day', ascending=False)
topSoldPCs.columns = ['item_id', 'sold_pcs']
topSoldPCs = topSoldPCs.nlargest(20, ['sold_pcs'])
topSoldPCs['sold_pcs'] = topSoldPCs['sold_pcs'] = round(topSoldPCs['sold_pcs'] / 1000)
import plotly.graph_objects as go
fig = go.Figure(data=[go.Table(header=dict(values=list(topSoldPCs.columns), fill_color='light blue', align='left'), cells=dict(values=[topSoldPCs.item_id, topSoldPCs.sold_pcs], fill_color='orange', align='left'))])
fig.update_layout(title='The most sold items(thousand pcs) by item_id from 2013 January to 2015 October', yaxis_zeroline=False, xaxis_zeroline=False)
fig.show()
rankItems = dsize_sales[['item_category_id', 'revenue']]
rankItems['revenue'] = round(rankItems['revenue'])
topCatogries = rankItems.groupby(['item_category_id'], as_index=False).sum().sort_values('revenue')
topCatogries['revenue'] = round(topCatogries['revenue'] / 1000000)
top20Catogries = topCatogries.nlargest(20, ['revenue'])
Catogries = top20Catogries[['item_category_id', 'revenue']]
Catogries.head()
import plotly.graph_objects as go
df = px.data.tips()
labels = Catogries['item_category_id']
values = Catogries['revenue']
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
fig.update_traces(textposition='inside', textinfo='label+percent')
fig.update_layout(title_text='Top 20 best-selling categories by revenue from 2013 January to 2015 October (Ruple-millions)')
fig.show()
rankSoldCat = dsize_sales[['item_category_id', 'item_cnt_day']]
topSoldPCat = rankSoldCat.groupby(by=['item_category_id'], as_index=False).sum().sort_values('item_cnt_day', ascending=False)
topSoldPCat.columns = ['item_category_id', 'sold_pcs']
topSoldPCat = topSoldPCat.nlargest(20, ['sold_pcs'])
topSoldPCat['sold_pcs'] = topSoldPCat['sold_pcs'] = round(topSoldPCat['sold_pcs'] / 1000)
import plotly.graph_objects as go
fig = go.Figure(data=[go.Table(header=dict(values=list(topSoldPCat.columns), fill_color='purple', font=dict(color='white', size=12), align='left'), cells=dict(values=[topSoldPCat.item_category_id, topSoldPCat.sold_pcs], line_color='black', fill_color='rgba(152, 0, 0, .3)', align='left'))])
fig.update_layout(title='The most sold products(thousand pcs) by category_id from 2013 January to 2015 October', yaxis_zeroline=False, xaxis_zeroline=False)
fig.show()
topSoldPCat.tail()
dsize_sales['month'] = pd.to_datetime(dsize_sales['date']).dt.month
dsize_sales['year'] = pd.to_datetime(dsize_sales['date']).dt.year
newMonthSales = dsize_sales[['year', 'month', 'revenue']]
newMonthSales.head()
rankMonths = newMonthSales[['month', 'revenue']]
rankMonths['revenue'] = round(rankMonths['revenue'])
topMonths = rankMonths.groupby(['month']).sum().sort_values('revenue', ascending=False)
topMonths['revenue'] = round(topMonths['revenue'] / 1000000)
topMonths
plotTopFourShops = dsize_sales.loc[(dsize_sales['shop_id'] == 31) | (dsize_sales['shop_id'] == 25) | (dsize_sales['shop_id'] == 28) | (dsize_sales['shop_id'] == 42)]
plotTopFourShops.head()
plotTopFourShops = plotTopFourShops[['year', 'month', 'shop_id', 'revenue']]
sep = plotTopFourShops.groupby(['year', 'month', 'shop_id'], as_index=False).sum()
sep['revenue(m)'] = round(sep['revenue'] / 1000000)
sep.head()
shop25 = sep.loc[sep['shop_id'] == 25]
shop25 = shop25[['year', 'month', 'revenue(m)']]
shop25.columns = ['year', 'month', 'shop25']
shop25.head()
shop28 = sep.loc[sep['shop_id'] == 28]
shop28 = shop28[['year', 'month', 'revenue(m)']]
shop28.columns = ['year', 'month', 'shop28']
shop31 = sep.loc[sep['shop_id'] == 31]
shop31 = shop31[['year', 'month', 'revenue(m)']]
shop31.columns = ['year', 'month', 'shop31']
from functools import reduce
plotTopThreeShops = [shop25, shop28, shop31]
shop_final = reduce(lambda left, right: pd.merge(left, right, on=('year', 'month')), plotTopThreeShops).reset_index()
shop_final.head()
import plotly.graph_objects as go
fig = go.Figure()
x = ['2013-01', '2013-02', '2013-03', '2013-04', '2013-05', '2013-06', '2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12', '2014-01', '2014-02', '2014-03', '2014-04', '2014-05', '2014-06', '2014-07', '2014-08', '2014-09', '2014-10', '2014-11', '2014-12', '2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10']
fig.add_trace(go.Scatter(x=x, y=shop_final['shop25'], mode='lines+markers', name='shop25', line_color='Red'))
fig.add_trace(go.Scatter(x=x, y=shop_final['shop28'], mode='lines+markers', name='shop28', line_color='Green'))
fig.add_trace(go.Scatter(x=x, y=shop_final['shop31'], mode='lines+markers', name='shop31', line_color='Blue'))
fig.update_layout(title_text='Top three best selling shops from 2013 January to 2015 October', yaxis_title='Revenue in Ruble (millions)')
fig.show()
combinedTable = dsize_sales[['date', 'shop_id', 'item_id', 'item_cnt_day']]
combinedTable.head(20)
before_final = combinedTable.assign(month=combinedTable.date + pd.to_timedelta(1 - combinedTable.date.dt.day, 'D')).groupby(['month']).apply(lambda x: x.groupby(['shop_id', 'item_id']).sum()).reset_index()
before_final.head()
