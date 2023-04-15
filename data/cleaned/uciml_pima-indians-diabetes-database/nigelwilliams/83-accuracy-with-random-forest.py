import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
X = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
y = X.pop('Outcome')
X
numRows = 8
numCols = 2
figWidth = 5
figHeight = 5
(fig, axes) = plt.subplots(numRows, numCols, sharex=False, figsize=(numCols * figWidth, numRows * figHeight))
for i in range(numRows):
    colname = X.columns[i]
    var2plot = X[colname]
    sns.histplot(x=var2plot, hue=y, ax=axes[i][0], common_bins=False, element='step')
    sns.histplot(x=var2plot[var2plot > 0], hue=y, ax=axes[i][1], common_bins=False, element='step')
    plt.legend()
for i in range(1, 6):
    value_counts = X.iloc[:, i].value_counts().sort_index()
    print('{} \t {}'.format(X.columns[i], value_counts[0] / 768))
from sklearn.impute import SimpleImputer
start_i = 1
end_i = 5
my_imputer = SimpleImputer(missing_values=0, strategy='median')
imputed_cols = pd.DataFrame(my_imputer.fit_transform(X.iloc[:, start_i:end_i + 1]))
imputed_cols.columns = X.columns[start_i:end_i + 1]
X_i = pd.concat([X.Pregnancies, imputed_cols, X.iloc[:, 6:8]], axis=1)
X_i
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0)
my_X = pd.concat([X_i.iloc[:, :3], X_i.iloc[:, 5:]], axis=1)
(X_t, X_v, y_t, y_v) = train_test_split(my_X, y, train_size=0.8, test_size=0.2, random_state=0)