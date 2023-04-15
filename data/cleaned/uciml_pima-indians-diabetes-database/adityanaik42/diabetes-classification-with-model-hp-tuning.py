import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.info()
data.describe()
ax1 = sns.violinplot(x='Outcome', y='BloodPressure', data=data, figsize=(15, 10), palette='muted', split=True)
ax2 = sns.violinplot(x='Outcome', y='Glucose', data=data, figsize=(15, 10), palette='muted', split=True)
ax2 = sns.violinplot(x='Outcome', y='Pregnancies', data=data, figsize=(15, 10), palette='muted', split=True)
ax2 = sns.violinplot(x='Outcome', y='SkinThickness', data=data, figsize=(15, 10), palette='muted', split=True)
ax2 = sns.violinplot(x='Outcome', y='Insulin', data=data, figsize=(15, 10), palette='muted', split=True)
ax2 = sns.violinplot(x='Outcome', y='BMI', data=data, figsize=(15, 10), palette='muted', split=True)
ax2 = sns.violinplot(x='Outcome', y='DiabetesPedigreeFunction', data=data, figsize=(15, 10), palette='muted', split=True)
ax2 = sns.violinplot(x='Outcome', y='Age', data=data, figsize=(15, 10), palette='muted', split=True)
df1 = data[data['Outcome'] == 1]
df2 = data[data['Outcome'] == 0]
df1 = df1.replace({'BloodPressure': 0}, np.median(df1['BloodPressure']))
df2 = df2.replace({'BloodPressure': 0}, np.median(df2['BloodPressure']))
df1 = df1.replace({'Glucose': 0}, np.median(df1['Glucose']))
df2 = df2.replace({'Glucose': 0}, np.median(df2['Glucose']))
df1 = df1.replace({'SkinThickness': 0}, np.median(df1['SkinThickness']))
df2 = df2.replace({'SkinThickness': 0}, np.median(df2['SkinThickness']))
df1 = df1.replace({'Insulin': 0}, np.median(df1['Insulin']))
df2 = df2.replace({'Insulin': 0}, np.median(df2['Insulin']))
df1 = df1.replace({'BMI': 0}, np.median(df1['BMI']))
df2 = df2.replace({'BMI': 0}, np.median(df2['BMI']))
dataframe = [df1, df2]
data_clean = pd.concat(dataframe)
data_clean.head()
data_clean.describe()
X = data.drop(['Outcome', 'SkinThickness', 'BloodPressure', 'Insulin'], axis=1)
y = data['Outcome']
X.head()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
numerical_pipe = Pipeline([('std_scaler', StandardScaler())])
X_train_norm = numerical_pipe.fit_transform(X)
(X_train, X_test, y_train, y_test) = train_test_split(X_train_norm, y, test_size=0.2, random_state=42)

def plot_search_results(grid, params):
    """
    Parameters: 
        grid: A trained GridSearchCV object.
        params: A dictionary of model attributes as keys and the values as the list of values the parameters 
                take for the grid search 
    """
    results = grid.cv_results_
    means_test = results['mean_test_score']
    means_train = results['mean_train_score']
    masks = []
    masks_names = list(grid.best_params_.keys())
    for (p_k, p_v) in grid.best_params_.items():
        masks.append(list(results['param_' + p_k].data == p_v))
    pram_preformace_in_best = {}
    for (i, p) in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i + 1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[0][p])
        y_1 = np.array(means_test[best_index])
        y_2 = np.array(means_train[best_index])
        plt.title('Mean score per parameter')
        plt.ylabel('Mean accuracy score')
        plt.plot(x, y_1, linestyle='--', marker='o', label='Test')
        plt.plot(x, y_2, linestyle='-', marker='^', label='Train')
        plt.xlabel(p.upper())
        plt.legend()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
logistic_model = LogisticRegression()
param_grid = [{'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'max_iter': [15, 20, 30, 40]}, {'solver': ['lbfgs'], 'penalty': ['none', 'l2'], 'max_iter': [15, 20, 30, 40]}]
model1_grid_search = GridSearchCV(logistic_model, param_grid, cv=4, scoring='accuracy', return_train_score=True)