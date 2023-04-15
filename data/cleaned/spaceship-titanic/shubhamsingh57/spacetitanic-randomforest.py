import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.shape
df.head()
df.info()
df.isnull().sum().plot(kind='bar')
from sklearn.impute import SimpleImputer
Si = SimpleImputer(strategy='median')
cat_data = df.select_dtypes('object')