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
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data.head()
test_data.head()
print('Train_Data_Set --> Rows : {0} , Columns : {1}'.format(train_data.shape[0], train_data.shape[1]))
print('Test_Data_Set --> Rows : {0} , Columns : {1}'.format(test_data.shape[0], test_data.shape[1]))
train_data.info()
test_data.info()
df = train_data.dtypes.value_counts()
df.plot(kind='pie')
test_data.dtypes.value_counts().plot(kind='bar')
num_data = [i for i in train_data.select_dtypes(['int', 'float'])]
num_data
cat_data = [i for i in train_data.select_dtypes(exclude=['int', 'float'])]
cat_data
cat_test_data = [i for i in test_data.select_dtypes(exclude=['int', 'float'])]
cat_test_data
train_data.describe().T
train_data.isna().any()
train_data.isna().sum()
sns.heatmap(train_data.isna(), cmap='hot', cbar=False)
train_data.isna().sum() / len(train_data) * 100
for i in cat_data:
    train_data[i].fillna(train_data[i].value_counts().index[0], inplace=True)
for i in cat_test_data:
    test_data[i].fillna(test_data[i].value_counts().index[0], inplace=True)
for i in num_data:
    train_data[i].fillna(train_data[i].mean(), inplace=True)
sns.heatmap(train_data.isna(), cbar=False, cmap='coolwarm')
train_data.isna().any()
sns.heatmap(train_data.corr(), cmap='coolwarm', annot=True, linewidths=0.2)
fig = plt.figure(figsize=(12, 4))
for (i, col) in enumerate(num_data):
    ax = fig.add_subplot(3, 2, i + 1)
    sns.boxplot(x=train_data[col], ax=ax)
fig.tight_layout()
