import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, PrecisionRecallDisplay, RocCurveDisplay, precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.dtypes
target = data.Outcome.value_counts(normalize=True)
target.plot(kind='pie')

target
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
(train_index, test_index) = next(sss.split(data.drop('Outcome', axis=1), data['Outcome']))
X_train = data.loc[train_index,].drop('Outcome', axis=1)
y_train = data.loc[train_index]['Outcome']
X_test = data.loc[test_index,].drop('Outcome', axis=1)
y_test = data.loc[test_index]['Outcome']
lr = LogisticRegression(max_iter=500, C=100)