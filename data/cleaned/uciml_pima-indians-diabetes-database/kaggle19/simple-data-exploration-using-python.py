import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time

data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print('This dataset has {} samples with {} features each.'.format(data.shape[0], data.shape[1]))
print('\nDisplay the first 10 record')

print('Statistical description of dataset\n--------------------------------------')

print('Note\n-----')
print('All values are numerical')
print("'Outcome' is the target/label variable that can have only binary value(0/1)\n\n")
data.info()
n_records = data.shape[0]
n_with_diabetes = data[data['Outcome'] == 1].shape[0]
n_without_diabetes = data[data['Outcome'] == 0].shape[0]
greater_percent = n_with_diabetes * 100 / float(n_records)
print('\nTotal number of individuals: {}'.format(n_records))
print('Individuals with diabetes: {}'.format(n_with_diabetes))
print('Individuals without diabetes: {}'.format(n_without_diabetes))
print('Percentage of individuals with diabetes: {:.2f}%\n'.format(greater_percent))
sns.countplot(data['Outcome'], label='Count')
featurelist = []
count_of_zero_list = []
for col in data:
    cnt = 0
    for i in data[col]:
        if i == 0:
            cnt = cnt + 1
    if col != 'Outcome':
        featurelist.append(col)
        count_of_zero_list.append(cnt)
objects = tuple(featurelist)
y_pos = np.arange(len(featurelist))
performance = count_of_zero_list
fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 11
fig_size[1] = 3
plt.bar(y_pos, performance, align='center', color='b', alpha=0.9)
plt.xticks(y_pos, objects)
plt.ylabel('Count of 0 values')
plt.title('Count of 0 values per feature')
plt.grid(True)

data.hist(figsize=(10, 8), color='b')
pd.DataFrame.skew(data, axis=0)
(fig, axes) = plt.subplots(nrows=4, ncols=2, figsize=(16, 16))
data.boxplot(column='Pregnancies', by='Outcome', ax=axes[0, 0])
data.boxplot(column='Glucose', by='Outcome', ax=axes[0, 1])
data.boxplot(column='BloodPressure', by='Outcome', ax=axes[1, 0])
data.boxplot(column='SkinThickness', by='Outcome', ax=axes[1, 1])
data.boxplot(column='Insulin', by='Outcome', ax=axes[2, 0])
data.boxplot(column='BMI', by='Outcome', ax=axes[2, 1])
data.boxplot(column='DiabetesPedigreeFunction', by='Outcome', ax=axes[3, 0])
data.boxplot(column='Age', by='Outcome', ax=axes[3, 1])
fig.tight_layout()
corr = data.corr()
fig = plt.figure(figsize=(7, 5.5))
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot=True, mask=mask, vmax=0.5, linewidths=0.1)
fig.suptitle('Attribute Correlation Matrix', fontsize=14)