import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import norm, skew, kurtosis

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.svm import SVR
import catboost as cb
from sklearn.ensemble import StackingRegressor
from category_encoders import MEstimateEncoder
from xgboost import XGBRegressor
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
X = train_df.drop(columns=['SalePrice', 'Id'])
y = train_df['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.15, random_state=1994)
data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
X = data.drop(columns=['SalePrice', 'Id'])
y = data['SalePrice']
X_train_data = X[:1460]
X_test_data = X[1460:]
y_train_data = y[:1460]
y_test_data = y[1460:]
(X_train, X_test, y_train, y_test) = train_test_split(X_train_data, y_train_data, test_size=0.15, random_state=1994)
data[:1460].tail()
data[1460:].head()
(f, ax) = plt.subplots(figsize=(30, 25))
mat = train_df.corr('pearson')
mask = np.triu(np.ones_like(mat, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(mat, mask=mask, cmap=cmap, vmax=1, center=0, annot=True, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})

mat['SalePrice'].sort_values()