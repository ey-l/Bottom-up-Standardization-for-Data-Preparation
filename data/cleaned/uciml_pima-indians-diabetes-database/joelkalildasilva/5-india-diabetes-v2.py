import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head().T
df.columns


sns.pairplot(df, hue='Outcome')
df.shape
df.Outcome.value_counts()
df.describe()
data_train = df.sample(frac=0.8, random_state=1)
data_test = df.drop(data_train.index)
X_train = data_train.drop(['Outcome'], axis=1)
y_train = data_train['Outcome']
X_test = data_test.drop(['Outcome'], axis=1)
y_test = data_test['Outcome']
from sklearn import ensemble
from sklearn.metrics import accuracy_score, confusion_matrix
rf = ensemble.RandomForestClassifier()