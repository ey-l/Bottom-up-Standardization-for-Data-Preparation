import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
import seaborn as sns
import plotly.graph_objects as go
from sklearn import preprocessing
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.dtypes
print('Test statistics: \n', _input0.describe())
print('\n')
print('Train statistics: \n', _input1.describe())

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
_input0 = reduce_mem_usage(_input0)
traint = reduce_mem_usage(_input1)
missing_test = pd.DataFrame(_input0.isna().sum())
missing_test.sort_values(by=0, ascending=False)
missing_train = pd.DataFrame(_input1.isna().sum())
missing_train.sort_values(by=0, ascending=False)
fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Bar(y=missing_train[0], x=missing_train.index, marker=dict(color=[n for n in range(14)], coloraxis='coloraxis')), 1, 1)
fig.add_trace(go.Bar(y=missing_test[0], x=missing_test.index, marker=dict(color=[n for n in range(14)], coloraxis='coloraxis')), 1, 2)
fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False, title_text="Features' Null Value Distribution in Train and Test Data", title_x=0.5)
fig.show()
_input1 = _input1.replace(['', ' '], np.NaN)
_input0 = _input0.replace(['', ' '], np.NaN)
impmean = SimpleImputer(strategy='mean', missing_values=np.nan)
impcomm = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
impconst0 = SimpleImputer(strategy='constant', missing_values=np.nan, fill_value=0)
impconstf = SimpleImputer(strategy='constant', missing_values=np.nan, fill_value=False)
impconstx = SimpleImputer(strategy='constant', missing_values=np.nan, fill_value='Mr. XXXX')