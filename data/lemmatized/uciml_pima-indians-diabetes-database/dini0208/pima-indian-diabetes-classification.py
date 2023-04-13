import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data[['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.NaN)
data.head()
data_nan = data.isna().sum()
data_nan = pd.DataFrame(data_nan, columns=['NaN count'])
data_nan
data_nan = data_nan.reset_index()
pass
pass
for p in plot.patches:
    plot.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2.0, p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)
pass
pass
pass
pass
pass
data.info()
pass
pass
for p in plot_outcome.patches:
    plot_outcome.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2.0, p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)
pass
pass
pass
pass
pass
pass
pass
for p in plt_preg.patches:
    plt_preg.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2.0, p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)
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
df1 = data[data['Outcome'] == 0]
pass
df2 = data[data['Outcome'] == 1]
pass
pass
pass
pass
pass
pass
pass
pass
df1 = data[data['Outcome'] == 0]
pass
df2 = data[data['Outcome'] == 1]
pass
pass
pass
pass
pass
pass
pass
pass
df1 = data[data['Outcome'] == 0]
pass
df2 = data[data['Outcome'] == 1]
pass
pass
pass
pass
pass
pass
pass
pass
df1 = data[data['Outcome'] == 0]
pass
df2 = data[data['Outcome'] == 1]
pass
pass
pass
pass
pass
pass
pass
pass
df1 = data[data['Outcome'] == 0]
pass
df2 = data[data['Outcome'] == 1]
pass
pass
pass
pass
pass
pass
pass
pass
df1 = data[data['Outcome'] == 0]
pass
df2 = data[data['Outcome'] == 1]
pass
pass
pass
pass
pass
pass
pass
pass
df1 = data[data['Outcome'] == 0]
pass
df2 = data[data['Outcome'] == 1]
pass
pass
pass
pass
pass
pass
pass
pass
pass
for p in plt_preg.patches:
    plt_preg.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2.0, p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)
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
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2, weights='uniform')
data_imputed = imputer.fit_transform(data)
data_imputed = pd.DataFrame(data_imputed, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
data_imputed.head()
data_imputed['Outcome'].value_counts()
from sklearn.utils import resample
df_majority = data_imputed[data_imputed.Outcome == 0]
df_minority = data_imputed[data_imputed.Outcome == 1]
data_minority_upsampled = resample(df_minority, replace=True, n_samples=500, random_state=123)
data_upsampled = pd.concat([df_majority, data_minority_upsampled])
data_upsampled['Outcome'].value_counts()
x = data_upsampled.drop(columns=['Outcome'], axis=1)
y = data_upsampled.Outcome
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = StandardScaler().fit_transform(x)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
accuracy = pd.DataFrame(columns=['classifiers', 'accuracy', 'auc'])
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
alphas = np.linspace(1, 10, 100)
ridgeClassifiercv = LogisticRegressionCV(penalty='l2', Cs=1 / alphas, solver='liblinear')