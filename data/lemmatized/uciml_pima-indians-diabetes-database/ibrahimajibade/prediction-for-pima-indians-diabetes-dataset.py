import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.describe()
data.info()
pass
pass
pass
pass
pass
pass
pass
from sklearn.preprocessing import StandardScaler
scaled = StandardScaler()