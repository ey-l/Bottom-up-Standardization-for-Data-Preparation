import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head(10)
from matplotlib import pyplot as plt
import seaborn as sns
df.shape
df.Outcome.value_counts()
df.describe()
data_train = df.sample(frac=0.8, random_state=1)
data_test = df.drop(data_train.index)
X_train = data_train.drop(['Outcome'], axis=1)
y_train = data_train['Outcome']
X_test = data_test.drop(['Outcome'], axis=1)
y_test = data_test['Outcome']
plt.figure(figsize=(9, 9))
logistique = lambda x: np.exp(x) / (1 + np.exp(x))
x_range = np.linspace(-10, 10, 50)
y_values = logistique(x_range)
plt.plot(x_range, y_values, color='red')
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')