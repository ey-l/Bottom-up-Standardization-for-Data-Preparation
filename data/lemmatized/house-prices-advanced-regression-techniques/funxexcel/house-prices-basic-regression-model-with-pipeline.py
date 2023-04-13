import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
y = _input1['SalePrice']
X = _input1.drop(['SalePrice', 'Id'], axis=1)
(X_train, X_test, y_train, y_test) = model_selection.train_test_split(X, y, test_size=0.2, random_state=200)
numerical_features = [c for (c, dtype) in zip(X.columns, X.dtypes) if dtype.kind in ['i', 'f']]
print('Numerical : ' + str(numerical_features))
preprocessor = make_column_transformer((make_pipeline(KNNImputer(n_neighbors=10), KBinsDiscretizer(n_bins=6), SelectKBest(chi2, k=15)), numerical_features))
regModel = make_pipeline(preprocessor, LinearRegression())