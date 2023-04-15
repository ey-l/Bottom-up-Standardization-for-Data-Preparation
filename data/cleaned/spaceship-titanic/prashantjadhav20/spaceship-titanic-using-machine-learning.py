import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
train_dataset = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_dataset
train_dataset.info()
train_dataset.shape
train_dataset.isnull().sum()
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
for col in train_dataset.columns:
    if train_dataset[col].dtype == object:
        train_dataset[col] = train_dataset[col].astype(str)
        d1 = train_dataset[col].unique()
        d2 = d1[-2]
        train_dataset[col] = train_dataset[col].fillna(d2)
        train_dataset[col] = l1.fit_transform(train_dataset[col])
    elif train_dataset[col].dtype == float or train_dataset[col].dtype == int:
        m1 = train_dataset[col].mean()
        m1 = round(m1)
        train_dataset[col] = train_dataset[col].fillna(m1)
print(train_dataset)
train_dataset.info()
train_dataset['Transported'] = l1.fit_transform(train_dataset['Transported'])
train_dataset['Transported']
x = train_dataset.iloc[:, :-1].values
y = train_dataset.iloc[:, -1].values
y
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
x_scale = std.fit_transform(x)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x_scale, y, test_size=0.2, random_state=11)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()