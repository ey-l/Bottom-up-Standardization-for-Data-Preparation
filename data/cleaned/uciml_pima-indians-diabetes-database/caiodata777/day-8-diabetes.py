import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df.isnull().sum()
dict = {}
for i in list(df.columns):
    dict[i] = df[i].value_counts().shape[0]
pd.DataFrame(dict, index=['unique count']).transpose()
df['Pregnancies'].value_counts()
df = df.groupby('Pregnancies').filter(lambda x: len(x) > 8)
df['Pregnancies'].value_counts()
(x_train, x_test, y_train, y_test) = train_test_split(df.drop('Outcome', axis=1), df['Outcome'], test_size=0.25, random_state=0)
pipe1 = Pipeline([('scaler', StandardScaler()), ('model', SVC(random_state=0))])