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
plt.figure(figsize=(12, 8))
plot = sns.barplot(x='index', y='NaN count', data=data_nan, palette='rocket')
for p in plot.patches:
    plot.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2.0, p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)
plt.xticks(fontsize=12, rotation=40)
plt.xlabel('Variable', fontsize=15)
plt.yticks(fontsize=12)
plt.ylabel('NaN Count', fontsize=15)
plt.title('NaN Count of variables', fontsize=20)

data.info()
plt.figure(figsize=(12, 8))
plot_outcome = sns.countplot(x='Outcome', data=data, palette='husl')
for p in plot_outcome.patches:
    plot_outcome.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2.0, p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)
plt.title('Count of Outcome', fontsize=20)
plt.xlabel('Outcome', fontsize=15)
plt.xticks(np.arange(2), ('No', 'Yes'), fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Count', fontsize=15)

plt.figure(figsize=(12, 8))
plt_preg = sns.countplot(x='Pregnancies', data=data, palette='husl')
for p in plt_preg.patches:
    plt_preg.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2.0, p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)
plt.title('Count of Number of Pregnancies', fontsize=20)
plt.xlabel('Number of Pregnancies', fontsize=15)
plt.xticks(np.arange(18), fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Count', fontsize=15)

plt.figure(figsize=(12, 8))
sns.distplot(data['Glucose'], kde=True, color='Orange')
plt.title('Histogram of Glucose', fontsize=20)
plt.xlabel('Glucose Level', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(figsize=(12, 8))
sns.distplot(data['BloodPressure'], kde=True, color='Purple')
plt.title('Histogram of BloodPressure', fontsize=20)
plt.xlabel('BloodPressure Level', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(figsize=(12, 8))
sns.distplot(data['SkinThickness'], kde=True, color='Red')
plt.title('Histogram of SkinThickness', fontsize=20)
plt.xlabel('SkinThickness Level', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(figsize=(12, 8))
sns.distplot(data['Insulin'], kde=True, color='Orange')
plt.title('Histogram of Insulin', fontsize=20)
plt.xlabel('Insulin Level', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(figsize=(12, 8))
sns.distplot(data['BMI'], kde=True, color='Blue')
plt.title('Histogram of BMI', fontsize=20)
plt.xlabel('BMI Value', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(figsize=(12, 8))
sns.distplot(data['DiabetesPedigreeFunction'], kde=True, color='Brown')
plt.title('Histogram of Diabetes Pedigree Function', fontsize=20)
plt.xlabel('Diabetes Pedigree Function Value', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(figsize=(12, 8))
sns.distplot(data['Age'], kde=True, color='Black')
plt.title('Histogram of Age', fontsize=20)
plt.xlabel('Age in years', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(figsize=(12, 8))
df1 = data[data['Outcome'] == 0]
sns.distplot(df1['Age'], kde=True, label='No', color='red')
df2 = data[data['Outcome'] == 1]
sns.distplot(df2['Age'], kde=True, label='Yes', color='blue')
plt.legend(prop={'size': 12})
plt.title('Age Vs. Outcome', fontsize=20)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(figsize=(12, 8))
df1 = data[data['Outcome'] == 0]
sns.distplot(df1['DiabetesPedigreeFunction'], kde=True, label='No', color='green')
df2 = data[data['Outcome'] == 1]
sns.distplot(df2['DiabetesPedigreeFunction'], kde=True, label='Yes', color='orange')
plt.legend(prop={'size': 12})
plt.title('Diabetes Pedigree Function Vs. Outcome', fontsize=20)
plt.xlabel('Diabetes Pedigree Function Value', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(figsize=(12, 8))
df1 = data[data['Outcome'] == 0]
sns.distplot(df1['BMI'], kde=True, label='No', color='purple')
df2 = data[data['Outcome'] == 1]
sns.distplot(df2['BMI'], kde=True, label='Yes', color='red')
plt.legend(prop={'size': 12})
plt.title('BMI Vs. Outcome', fontsize=20)
plt.xlabel('BMI Value', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(figsize=(12, 8))
df1 = data[data['Outcome'] == 0]
sns.distplot(df1['Insulin'], kde=True, label='No', color='black')
df2 = data[data['Outcome'] == 1]
sns.distplot(df2['Insulin'], kde=True, label='Yes', color='yellow')
plt.legend(prop={'size': 12})
plt.title('Insulin Vs. Outcome', fontsize=20)
plt.xlabel('Insulin Level', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(figsize=(12, 8))
df1 = data[data['Outcome'] == 0]
sns.distplot(df1['SkinThickness'], kde=True, label='No', color='red')
df2 = data[data['Outcome'] == 1]
sns.distplot(df2['SkinThickness'], kde=True, label='Yes', color='green')
plt.legend(prop={'size': 12})
plt.title('SkinThickness Vs. Outcome', fontsize=20)
plt.xlabel('SkinThickness', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(figsize=(12, 8))
df1 = data[data['Outcome'] == 0]
sns.distplot(df1['BloodPressure'], kde=True, label='No', color='crimson')
df2 = data[data['Outcome'] == 1]
sns.distplot(df2['BloodPressure'], kde=True, label='Yes', color='gold')
plt.legend(prop={'size': 12})
plt.title('Blood Pressure level Vs. Outcome', fontsize=20)
plt.xlabel('Blood Pressure Level', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(figsize=(12, 8))
df1 = data[data['Outcome'] == 0]
sns.distplot(df1['Glucose'], kde=True, label='No', color='teal')
df2 = data[data['Outcome'] == 1]
sns.distplot(df2['Glucose'], kde=True, label='Yes', color='peru')
plt.legend(prop={'size': 12})
plt.title('Glucose level Vs. Outcome', fontsize=20)
plt.xlabel('Glucose Level', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(figsize=(12, 8))
plt_preg = sns.countplot(x='Pregnancies', data=data, palette='rocket', hue='Outcome')
for p in plt_preg.patches:
    plt_preg.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2.0, p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)
plt.title('Number of Pregnancies Vs. Outcome', fontsize=20)
plt.xlabel('Number of Pregnancies', fontsize=15)
plt.xticks(np.arange(18), fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.legend(['No', 'Yes'], loc='upper right', fontsize=15)

plt.subplots(figsize=(12, 8))
sns.set(font_scale=1.5)
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.title('Correlation Plot', fontsize=20)

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