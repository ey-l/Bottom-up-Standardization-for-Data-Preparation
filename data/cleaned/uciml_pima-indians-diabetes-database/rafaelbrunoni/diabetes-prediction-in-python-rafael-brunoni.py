import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import graphviz
import pydotplus
from plotly import tools
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, accuracy_score, auc
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(df)
print(df.describe())
col = ['n_pregnant', 'glucose_conc', 'bp', 'skin_len', 'insulin', 'bmi', 'pedigree_fun', 'age', 'Output']
df.columns = col
df.head()
(fig, ax) = plt.subplots(4, 2, figsize=(16, 16))
sns.distplot(df.age, bins=20, ax=ax[0, 0])
sns.distplot(df.n_pregnant, bins=20, ax=ax[0, 1])
sns.distplot(df.glucose_conc, bins=20, ax=ax[1, 0])
sns.distplot(df.bp, bins=20, ax=ax[1, 1])
sns.distplot(df.skin_len, bins=20, ax=ax[2, 0])
sns.distplot(df.insulin, bins=20, ax=ax[2, 1])
sns.distplot(df.pedigree_fun, bins=20, ax=ax[3, 0])
sns.distplot(df.bmi, bins=20, ax=ax[3, 1])
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=0)
corr = df.corr()
sns.set(font_scale=1.15)
plt.figure(figsize=(14, 10))
sns.heatmap(corr, vmax=0.8, linewidths=0.01, square=True, annot=True, cmap='YlGnBu', linecolor='black')
plt.title('Correlation between features')
DT = DecisionTreeClassifier()