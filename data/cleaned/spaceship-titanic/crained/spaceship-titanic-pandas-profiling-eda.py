import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train.head()
train.shape
total = train.isnull().sum().sort_values(ascending=False)
total
train.dtypes
import pandas_profiling
pandas_profiling.ProfileReport(train)