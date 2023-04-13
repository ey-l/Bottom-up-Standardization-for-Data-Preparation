import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.info()
y = _input1.SalePrice
df1 = _input1.drop(columns={'MiscFeature', 'Fence', 'PoolQC', 'Alley', 'SalePrice'}, axis=1)
df1.info()
import numpy as np
from sklearn.impute import SimpleImputer

def fit_missing_values(column):
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')