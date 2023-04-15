import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
df_train = pd.read_csv('data/input/digit-recognizer/train.csv')
df_train.describe()
df_train_x = df_train.drop('label', axis=1)
df_train_y = df_train[['label']]