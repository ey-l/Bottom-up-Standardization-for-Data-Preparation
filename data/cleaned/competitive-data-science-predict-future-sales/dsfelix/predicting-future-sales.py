import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_style('whitegrid')
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
item_categories_dict = dict(zip(item_categories['item_category_id'], item_categories['item_category_name']))
item_categories_items_list = [item_categories_dict[x] for x in items['item_category_id']]
items.insert(2, 'item_category_name', item_categories_items_list)
shops_dict = dict(zip(shops['shop_id'], shops['shop_name']))
shops_sales_train_list = [shops_dict[x] for x in sales_train['shop_id']]
sales_train.insert(3, 'shop_name', shops_sales_train_list)
items_dict = dict(zip(items['item_id'], items['item_name']))
items_sales_train_list = [items_dict[x] for x in sales_train['item_id']]
sales_train.insert(5, 'item_name', items_sales_train_list)
items_category_id_dict = dict(zip(items['item_id'], items['item_category_id']))
items_category_id_sales_train_list = [items_category_id_dict[x] for x in sales_train['item_id']]
sales_train.insert(7, 'item_category_id', items_category_id_sales_train_list)
items_category_name_dict = dict(zip(items['item_category_id'], items['item_category_name']))
items_category_name_sales_train_list = [items_category_name_dict[x] for x in sales_train['item_category_id']]
sales_train.insert(8, 'item_category_name', items_category_name_sales_train_list)
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test.head()
full_train_df = sales_train.copy()
features_to_drop = ['date', 'shop_name', 'item_name', 'item_price', 'item_category_id', 'item_category_name']
train_df = full_train_df.drop(features_to_drop, axis=1).copy()
train_df
train_df.dtypes
train_df['item_cnt_day'] = train_df['item_cnt_day'].astype(np.int64)
train_df.dtypes
train_df.describe()
multiply_by_minus_one = lambda x: x * -1 if x < 0 else x
train_df['item_cnt_day'] = train_df['item_cnt_day'].apply(multiply_by_minus_one)
train_df.describe()
train_df.corr()
train_df.hist(bins=15, figsize=(15, 10))
plt.title('Shop x Number of Products Sold')
sns.scatterplot(data=train_df, x=train_df['shop_id'], y=train_df['item_cnt_day'])

plt.title('Item x Number of Products Sold')
sns.scatterplot(data=train_df, x=train_df['item_id'], y=train_df['item_cnt_day'])

plt.title('Month x Number of Products Sold')
sns.scatterplot(data=train_df, x=train_df['date_block_num'], y=train_df['item_cnt_day'])

train_df.head()
grouped_train_df = train_df.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False).sum('item_cnt_day').copy()
grouped_train_df.head()
X = grouped_train_df.copy()
y = X.pop('item_cnt_day')
from sklearn.feature_selection import mutual_info_regression
discrete_features = X.dtypes == int
continuous_features = X.dtypes == float
mi_scores_discrete_features = mutual_info_regression(X[0:500000], y[0:500000], discrete_features=discrete_features, random_state=0)
mi_scores_discrete_features = pd.Series(mi_scores_discrete_features, name='MI Scores', index=X.columns)
mi_scores_discrete_features = mi_scores_discrete_features.sort_values(ascending=False)

def plot_mi_scores(scores):
    """
    Plots Mutual Information Scores in Ascending Order
    """
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title('Mutual Information Scores')
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores_discrete_features)
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)