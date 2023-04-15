import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
plt.style.use('fivethirtyeight')
diab = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diab.head()
diab.isnull().sum()
sns.countplot(x='Outcome', data=diab)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
X = diab[diab.columns[:8]]
Y = diab['Outcome']