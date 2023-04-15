import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head(5)
df.isnull()
df.fillna(0)
df.corr()
df.style.background_gradient(cmap='viridis')
import matplotlib.pyplot as plt
plt.imshow(df, cmap='RdYlBu')
import seaborn as sns
sns.heatmap(df, cmap='RdYlBu', linewidths=0.3, annot=True)
df1 = df.corr()
sns.heatmap(df1, cmap='RdYlBu', linewidths=0.5, annot=True)
df.describe()
col = df.columns.values
col
df.info()
col_value = df.values
X = col_value[:, :-1]
Y = col_value[:, -1]
print(X)
print(Y)
x = df.iloc[:, :8]
x.columns
y = df.iloc[:, -1]
y
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = np.array(x)
Y = np.array(y)
X_updated = X.reshape(len(X), -1)
(xtrain, xtest, ytrain, ytest) = train_test_split(X_updated, Y, random_state=10, test_size=0.2)
(xtrain.shape, xtest.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
lg = LogisticRegression(C=0.1)