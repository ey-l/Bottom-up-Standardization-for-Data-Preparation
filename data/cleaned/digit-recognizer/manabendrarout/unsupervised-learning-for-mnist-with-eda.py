import warnings
import sklearn.exceptions
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=sklearn.exceptions.UndefinedMetricWarning)
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import os
import time
import random
from scipy.stats import mode
import matplotlib.pyplot as plt
from matplotlib import offsetbox

import seaborn as sns
sns.set(style='whitegrid')
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
RANDOM_SEED = 42

def seed_everything(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
seed_everything()
data_path = 'data/input/digit-recognizer'
train_file_path = os.path.join(data_path, 'train.csv')
sample_sub_path = os.path.join(data_path, 'sample_submission.csv')
test_file_path = os.path.join(data_path, 'test.csv')
print(f'Training File path: {train_file_path}')
print(f'Sample Submission File path: {sample_sub_path}')
print(f'Test Files path: {test_file_path}')
train_df = pd.read_csv(train_file_path)
sample_sub_df = pd.read_csv(sample_sub_path)
test_df = pd.read_csv(test_file_path)
train_data = train_df.drop(['label'], axis=1).values.reshape(-1, 28, 28, 1)
train_data = test_df.values.reshape(-1, 28, 28, 1)
num_examples = 10
plt.figure(figsize=(20, 20))
for i in range(num_examples):
    plt.subplot(1, num_examples, i + 1)
    plt.imshow(train_data[i], cmap='Greys')
    plt.axis('off')

ax = plt.subplots(figsize=(18, 6))
sns.set_style('whitegrid')
sns.countplot(x='label', data=train_df)
plt.ylabel('No. of Observations', size=20)
plt.xlabel('Class Name', size=20)
iso = Isomap(n_components=2)