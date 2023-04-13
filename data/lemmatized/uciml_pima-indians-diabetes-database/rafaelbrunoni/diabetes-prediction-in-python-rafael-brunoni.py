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
pass
pass
pass
pass
pass
pass
pass
pass
pass
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
pass
pass
pass
pass
DT = DecisionTreeClassifier()