import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
sample_submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv', index_col='Id')
with open('_data/input/house-prices-advanced-regression-techniques/data_description.txt') as f:
    data_description = f.readlines()



textfile = open('data_description.txt', 'w')
for element in data_description:
    textfile.write(element + '\n')
textfile.close()
train.head()
train.info()
test.info()
train.describe()
test.describe()
PROJECT_ROOT_DIR = '.'
CHAPTER_ID = 'end_to_end_project'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, 'images', CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + '.' + fig_extension)
    print('Saving figure', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import matplotlib.pyplot as plt
train.hist(bins=50, figsize=(20, 15))
save_fig('attribute_histogram_plots')

train.plot(kind='scatter', figsize=(30, 20), x='Neighborhood', y='SalePrice', alpha=0.4)
plt.legend()
save_fig('housing_prices_scatterplot')
corr_matrix = train.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
scatter_matrix(train[attributes], figsize=(20, 10))
save_fig('scatter_matrix_plot')
housing = train.copy()
housing_tr = housing.copy()
housing_tr.head()
housing_labels = housing_tr['SalePrice'].copy()
housing_tr = housing_tr.drop('SalePrice', axis=1)
sample_incomplete_rows = housing_tr[housing_tr.isnull().any(axis=1)]
sample_incomplete_rows.info()
housing_num = housing_tr.select_dtypes(include=[np.number])
housing_cat = housing_tr.select_dtypes(include=['object'])
housing_num.head()
housing_cat.head()
housing_cat.columns.values
list(housing_cat)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')