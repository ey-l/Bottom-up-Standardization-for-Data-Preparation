import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix
diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes.columns
random.seed(0)
diabetes['BloodPressure'] = diabetes['BloodPressure'].replace(0, random.choice(diabetes[diabetes.BloodPressure != 0]['BloodPressure'].tolist()))
diabetes['SkinThickness'] = diabetes['SkinThickness'].replace(0, random.choice(diabetes[diabetes.SkinThickness != 0]['SkinThickness'].tolist()))
diabetes['Insulin'] = diabetes['Insulin'].replace(0, random.choice(diabetes[diabetes.Insulin != 0]['Insulin'].tolist()))
diabetes['Insulin_group'] = pd.cut(diabetes['Insulin'], bins=3, labels=False)
diabetes['BMI_group'] = pd.cut(diabetes['BMI'], bins=5, labels=False)
diabetes['pedigree_group'] = pd.cut(diabetes['DiabetesPedigreeFunction'], bins=3, labels=False)
diabetes = diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin_group', 'BMI_group', 'pedigree_group', 'Outcome']]
sns.heatmap(diabetes.iloc[:, :-1].corr())
plt.xticks(rotation=45)
plt.title('Heapmap of Selected Feautures')
X = diabetes.drop(columns=['Outcome'])
X = sm.add_constant(X)
y = diabetes['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=0)
log_reg = LogisticRegression()
penalty = ['l1', 'l2']
C = np.linspace(0, 5, 10)
hyperparameters = dict(C=C, penalty=penalty)
clf = GridSearchCV(log_reg, hyperparameters, cv=5, verbose=0)