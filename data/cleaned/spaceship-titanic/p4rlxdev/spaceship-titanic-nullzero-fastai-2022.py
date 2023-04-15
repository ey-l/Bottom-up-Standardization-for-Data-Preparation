import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from fastai.tabular.all import *
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.float_format = '{:.2f}'.format
path = Path('data/input/spaceship-titanic')
print(path)
df_train = pd.read_csv(path / 'train.csv')
df_test = pd.read_csv(path / 'test.csv')
df_test.head()
df_train.head()
df_train['HomePlanet'].unique()
mean = df_train['Age'].mean()
mean
df_train['Age'].fillna(mean, inplace=True)
df_train.isna().sum()
df_train.head()
splits = RandomSplitter()(df_train)
dls = TabularPandas(df_train, splits=splits, procs=[Categorify, FillMissing, Normalize], cat_names=['PassengerId', 'Cabin'], cont_names=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa'], y_names='Transported', y_block=CategoryBlock()).dataloaders(path='.')
learn = tabular_learner(dls, metrics=accuracy, layers=[10, 10])
learn.lr_find(suggest_funcs=(slide, valley))