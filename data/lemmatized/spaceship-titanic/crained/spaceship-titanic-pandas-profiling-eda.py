import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.shape
total = _input1.isnull().sum().sort_values(ascending=False)
total
_input1.dtypes
import pandas_profiling
pandas_profiling.ProfileReport(_input1)