import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from scipy.stats import skew
import plotly
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)
from sklearn.neighbors import KNeighborsRegressor
custom_colors = ['#B69BC5', '#BB1C8B', '#05A4C0', '#CCEBC5', '#D2A7D8', '#FDDAEC', '#85CEDA']
customPalette = sns.set_palette(sns.color_palette(custom_colors))
sns.palplot(sns.color_palette(custom_colors), size=1)
plt.tick_params(axis='both', labelsize=0, length=0)
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train_df.head()
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test_df.head()
train_df.info()