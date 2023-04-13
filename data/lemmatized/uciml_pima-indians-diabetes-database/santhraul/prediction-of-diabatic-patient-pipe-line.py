import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
df = data
from matplotlib import pyplot as plt
import seaborn as sns
pass
X = df.drop('Outcome', axis=1)
y = df[['Outcome']]
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()