import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
item_categ = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
item_categ.head()
item_categ['item_category_name'].unique()
item_categ['item_category_name'].nunique()
item_categ['item_category_name'] = label_encoder.fit_transform(item_categ['item_category_name'])
X = item_categ.iloc[:, [0, 1]].values
item = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item.head()
item['item_name'].unique()
item['item_name'].describe()
item['item_name'].nunique()
item['item_name'] = label_encoder.fit_transform(item['item_name'])
item.tail()
item['item_category_id'].unique()
item['item_category_id'].nunique()
X_ = item.iloc[:, [1, 2]].values
X_
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)