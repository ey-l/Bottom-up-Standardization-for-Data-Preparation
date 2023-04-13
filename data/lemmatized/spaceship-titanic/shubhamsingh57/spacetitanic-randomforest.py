import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.shape
_input1.head()
_input1.info()
_input1.isnull().sum().plot(kind='bar')
from sklearn.impute import SimpleImputer
Si = SimpleImputer(strategy='median')
cat_data = _input1.select_dtypes('object')