import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
demo = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
demo.head()
from matplotlib import pyplot as plt
import seaborn as sns
sns.pairplot(demo, hue='Outcome', diag_kind='kde')

import warnings
warnings.filterwarnings('ignore')
X = demo.drop('Outcome', axis=1)
y = demo[['Outcome']]
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()