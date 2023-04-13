import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedShuffleSplit
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data_copy = data.copy()
data.head()
outcome_values = data['Outcome'].value_counts()
outcome_values.sort_index()
pass
pass
data.info()
data.describe()
pass
corr = data.corr()
pass
pass
for (i, feature) in enumerate(data.columns):
    pass
pass
pass
missing_cols = ['Glucose', 'Insulin', 'SkinThickness', 'BloodPressure', 'BMI']
missing_counts = {}
total_rows = data.shape[0]
for col in missing_cols:
    count = (data[col] == 0).sum()
    missing_counts[col] = count
pass
pass
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 3, '{:1.2f}%'.format(height * 100 / total_rows), ha='center')
pass
for col in missing_cols:
    data.loc[data[col] == 0.0, [col]] = None
pass
for (i, col) in enumerate(missing_cols):
    pass
ax[1][2].set_visible(False)
for col in missing_cols:
    positive_median = data[data['Outcome'] == 1][col].median()
    negative_median = data[data['Outcome'] == 0][col].median()
    data.loc[(data['Outcome'] == 0) & data[col].isna(), col] = negative_median
    data.loc[(data['Outcome'] == 1) & data[col].isna(), col] = positive_median
pass
pass
pass
pass
data['CategoricalAge'] = pd.qcut(data['Age'], q=5)
data['CategoricalBMI'] = pd.qcut(data['BMI'], q=5)
data['CategoricalPregnancies'] = pd.qcut(data['Pregnancies'], q=5)
pass
pass
pass
pass
data = pd.get_dummies(data)
data.drop(['Age', 'Pregnancies', 'BMI'], axis=1, inplace=True)
continuous_variables = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction']
pass
pass
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[continuous_variables] = pd.DataFrame(scaler.fit_transform(data[continuous_variables]))
data_x = data.drop(['Outcome'], axis=1)
data_y = data['Outcome']
sss = StratifiedShuffleSplit(test_size=0.3, n_splits=1, random_state=4321)
(train_val_index, test_index) = next(sss.split(data_x, data_y))
(X_train, X_test) = (data_x.iloc[train_val_index, :], data_x.iloc[test_index])
(y_train, y_test) = (data_y[train_val_index], data_y[test_index])
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
scores = {}
models = [LogisticRegression(max_iter=10000), KNeighborsClassifier(), RandomForestClassifier(random_state=42), GradientBoostingClassifier(random_state=42)]
for model in models:
    cv_scores = cross_val_score(model, X_train, y_train)
    estimator = model.__class__.__name__
    scores[estimator] = np.mean(cv_scores) * 100
pass
for p in ax.patches:
    width = p.get_width()
    ax.text(width / 2, p.get_y() + 0.5, '{:1.2f}%'.format(width))
params = {'n_estimators': np.arange(100, 1001, 100), 'max_depth': np.arange(2, 41, 2)}