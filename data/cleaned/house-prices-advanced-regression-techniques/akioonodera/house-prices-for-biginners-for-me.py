import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sample_submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
f = open('_data/input/house-prices-advanced-regression-techniques/data_description.txt', 'r')
data_description = f.read()
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train = pd.DataFrame(train)
df_test = pd.DataFrame(test)
pd.set_option('display.max_columns', 100)
df_all = pd.concat([df_train.drop(columns='SalePrice'), df_test], ignore_index=True)
df_all
import missingno as msno
msno.matrix(df=df_all, figsize=(20, 14), color=(0, 0.3, 0.3))
from sklearn.preprocessing import LabelEncoder
for i in range(df_all.shape[1]):
    if df_all.iloc[:, i].dtypes == object:
        lbl = LabelEncoder()