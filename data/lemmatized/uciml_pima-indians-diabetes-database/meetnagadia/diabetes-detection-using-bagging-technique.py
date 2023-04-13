import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.columns
data.shape
data.head()
data.tail()
data.isnull().sum()
pass
pass
data.Outcome.value_counts()
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
X = data.drop('Outcome', axis=1)
y = data['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
kfold = model_selection.KFold(n_splits=3)
base_cls = DecisionTreeClassifier()
num_trees = 200
model = BaggingClassifier(base_estimator=base_cls, n_estimators=num_trees)