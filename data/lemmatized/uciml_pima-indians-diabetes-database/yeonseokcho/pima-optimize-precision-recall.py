import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import sys
sys.version
import sklearn
sklearn.__version__
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
diabetes_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_data
diabetes_data.Outcome.value_counts()
diabetes_data.describe()
import matplotlib.pyplot as plt
diabetes_data.hist(bins=20, figsize=(15, 7))
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
diabetes_data[zero_features] = diabetes_data[zero_features].replace(0, diabetes_data[zero_features].mean())
diabetes_data.describe()
import matplotlib.pyplot as plt
diabetes_data.hist(bins=20, figsize=(15, 7))
X = diabetes_data.drop('Outcome', axis=1)
X
y = diabetes_data.Outcome
y
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
(X_train, X_test, y_train, y_test) = train_test_split(X_scaled, y, test_size=0.2, random_state=2208, stratify=y)
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
import warnings
warnings.filterwarnings('ignore')
lr_clf = LogisticRegression(random_state=2208)