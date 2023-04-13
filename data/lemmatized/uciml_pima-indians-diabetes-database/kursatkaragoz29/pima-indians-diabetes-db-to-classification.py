from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.shape
data.info()
data.head()
data.describe()
data.describe().T
data.isnull().sum()
print('Number of rows with Glucose == 0 => {}'.format((data.Glucose == 0).sum()))
print('Number of rows with BloodPressure == 0 => {}'.format((data.BloodPressure == 0).sum()))
print('Number of rows with SkinThickness == 0 => {}'.format((data.SkinThickness == 0).sum()))
print('Number of rows with Insulin == 0 => {}'.format((data.Insulin == 0).sum()))
print('Number of rows with BMI == 0 => {}'.format((data.BMI == 0).sum()))
data_copy = data.copy(deep=True)
data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(data_copy.isnull().sum())
data_copy.hist(figsize=(15, 15))
data_copy.BloodPressure.fillna(data_copy.BloodPressure.mean(), inplace=True)
data_copy.Glucose.fillna(data_copy.Glucose.mean(), inplace=True)
data_copy.SkinThickness.fillna(data_copy.SkinThickness.median(), inplace=True)
data_copy.Insulin.fillna(data_copy.Insulin.median(), inplace=True)
data_copy.BMI.fillna(data_copy.BMI.median(), inplace=True)
data_copy.describe().T
data_copy.hist(figsize=(15, 15))
data_copy.head()
y = data_copy['Outcome'].values.reshape(-1, 1)
x_data = data_copy.drop(['Outcome'], axis=1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
x.head()
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=29)
x_train.head()
y_train[:5]
from sklearn.neighbors import KNeighborsClassifier
train_score_list = []
test_score_list = []
for each in range(2, 31):
    knn = KNeighborsClassifier(n_neighbors=each)