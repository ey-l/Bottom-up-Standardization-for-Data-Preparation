import pandas as pd
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isnull().sum()
df.dtypes
import seaborn as sns
df.columns
df.shape
df.describe()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, auc, confusion_matrix, classification_report
X = df.drop('Outcome', axis=1)
y = df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=10)