import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.info()
import missingno as msno
ax = msno.bar(data, color='tab:blue')
ax.set_title('Missing data report for Pima Indian Diabates dataset')
pass
pass
import seaborn as sns
pass
pass
tx = ax.set_title('Distribution of data by classes')
from sklearn.model_selection import train_test_split
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
fimp_model = RandomForestClassifier()