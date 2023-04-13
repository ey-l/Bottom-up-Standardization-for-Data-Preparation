import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input0.head()
print('Train_Data_Set --> Rows : {0} , Columns : {1}'.format(_input1.shape[0], _input1.shape[1]))
print('Test_Data_Set --> Rows : {0} , Columns : {1}'.format(_input0.shape[0], _input0.shape[1]))
_input1.info()
_input0.info()
df = _input1.dtypes.value_counts()
df.plot(kind='pie')
_input0.dtypes.value_counts().plot(kind='bar')
num_data = [i for i in _input1.select_dtypes(['int', 'float'])]
num_data
cat_data = [i for i in _input1.select_dtypes(exclude=['int', 'float'])]
cat_data
cat_test_data = [i for i in _input0.select_dtypes(exclude=['int', 'float'])]
cat_test_data
_input1.describe().T
_input1.isna().any()
_input1.isna().sum()
sns.heatmap(_input1.isna(), cmap='hot', cbar=False)
_input1.isna().sum() / len(_input1) * 100
for i in cat_data:
    _input1[i] = _input1[i].fillna(_input1[i].value_counts().index[0], inplace=False)
for i in cat_test_data:
    _input0[i] = _input0[i].fillna(_input0[i].value_counts().index[0], inplace=False)
for i in num_data:
    _input1[i] = _input1[i].fillna(_input1[i].mean(), inplace=False)
sns.heatmap(_input1.isna(), cbar=False, cmap='coolwarm')
_input1.isna().any()
sns.heatmap(_input1.corr(), cmap='coolwarm', annot=True, linewidths=0.2)
fig = plt.figure(figsize=(12, 4))
for (i, col) in enumerate(num_data):
    ax = fig.add_subplot(3, 2, i + 1)
    sns.boxplot(x=_input1[col], ax=ax)
fig.tight_layout()