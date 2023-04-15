import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np
train_path = 'data/input/spaceship-titanic/train.csv'
test_path = 'data/input/spaceship-titanic/test.csv'
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
data = train_data.append(test_data)
data.head(4)
data.describe()
data.info()
data.shape
data.duplicated()
missing_values = [missing_values for missing_values in data if data[missing_values].isnull().sum() > 0]
missing_data = data[missing_values]
missing_data.head(3)
null_values = data.isnull().sum().sort_values(ascending=False)
null_values
from sklearn.impute import SimpleImputer
for values in missing_data.columns:
    if missing_data[values].dtypes == 'object' and missing_data[values].isnull().sum() > 0:
        Imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        cat_data = missing_data.select_dtypes(exclude=np.number)
        data[cat_data.columns] = Imputer.fit_transform(cat_data)
    if missing_data[values].dtypes in ['int32', 'int64', 'float64'] and missing_data[values].isnull().sum() > 0:
        Imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        num_data = missing_data.select_dtypes(include=np.number)
        data[num_data.columns] = Imputer.fit_transform(num_data)
data.isnull().sum()
data.head(3)
data_copy = data.copy()
data_copy = data.drop('PassengerId', axis=1)
data_copy.head(3)