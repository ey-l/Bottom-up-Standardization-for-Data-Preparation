import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.info()
df.head()
pass
pass
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()