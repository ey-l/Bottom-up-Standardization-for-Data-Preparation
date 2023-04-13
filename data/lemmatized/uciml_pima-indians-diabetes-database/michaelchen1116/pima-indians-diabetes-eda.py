import numpy as np
import pandas as pd
import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes.columns
diabetes.info()
diabetes['Outcome'].value_counts()
pass
fig.suptitle('Distribution of Pregnancies column')
df = pd.DataFrame(diabetes['Pregnancies'].value_counts())
df.plot(kind='bar', legend=None, ax=axes[0])
pass
df1 = pd.DataFrame(diabetes['Pregnancies'].value_counts())
df1.loc[8] = df1[df1.index >= 8].Pregnancies.sum()
df1.drop(index=[9, 10, 11, 12, 13, 14, 15, 17], inplace=True)
df1.plot(kind='bar', legend=None, ax=axes[1])
pass
cols = ['Before censoring', 'After censoring']
rows = ['Frequency']
for (ax, col) in zip(axes, cols):
    ax.set_title(col)
for (ax, row) in zip(axes, rows):
    pass
    pass
pass
pass
zero_glucose = diabetes[diabetes['Glucose'] == 0].shape[0]
zero_glucose_outcome = diabetes[diabetes['Glucose'] == 0].Outcome.mean()
nonzero_glucose_outcome = round(diabetes[diabetes['Glucose'] != 0].Outcome.mean(), 2)
print(f'There are {zero_glucose} records with glucose of zero. \nThe rows where glucose are zero have a mean outcome of {zero_glucose_outcome}\nThe rows where glucose are non-zero have a mean outcome of {nonzero_glucose_outcome}')
i = diabetes[diabetes['Glucose'] == 0].index
diabetes.drop(i, inplace=True)
pass
pass
pass
pass
pass
pass
pass
pass
(k2, p) = stats.normaltest(diabetes[diabetes['SkinThickness'] != 0].SkinThickness)
alpha = 0.05
print('p = {:g}'.format(p))
if p < alpha:
    print('Reject null')
else:
    print('Fail to reject null')
pass
pass
pass
zero_insulin = diabetes[diabetes['Insulin'] == 0].shape[0]
zero_insulin_percentage = round(zero_insulin / diabetes.shape[0], 2) * 100
print(f'{zero_insulin_percentage}% of the records with insulin of zero.')
pass
pass
i = diabetes[diabetes['BMI'] == 0].index
diabetes.drop(i, inplace=True)
pass
pass
pass
diabetes['DiabetesPedigreeFunction'] = np.where(diabetes['DiabetesPedigreeFunction'] > 1.0, 1.0, diabetes['DiabetesPedigreeFunction'])
pass
pass
pass
pass
pass
pass
no_bp = diabetes.BloodPressure == 0
no_skin = diabetes.SkinThickness == 0
no_insulin = diabetes.Insulin == 0
pass
pass
pass
axs[0, 0].set_title('Pregnancies')
axs[0, 0].legend()
axs[0, 0].xaxis.label.set_visible(False)
pass
pass
axs[0, 1].set_title('Glucose')
axs[0, 1].legend()
axs[0, 1].xaxis.label.set_visible(False)
pass
pass
axs[0, 2].set_title('SkinThickness')
axs[0, 2].legend()
axs[0, 2].xaxis.label.set_visible(False)
pass
pass
axs[1, 0].set_title('Insulin')
axs[1, 0].legend()
axs[1, 0].xaxis.label.set_visible(False)
pass
pass
axs[1, 1].set_title('BMI')
axs[1, 1].legend()
axs[1, 1].xaxis.label.set_visible(False)
pass
pass
axs[1, 2].set_title('DiabetesPedigreeFunction')
axs[1, 2].legend()
axs[1, 2].xaxis.label.set_visible(False)
pass
pass
axs[2, 0].set_title('Age')
axs[2, 0].legend()
axs[2, 0].xaxis.label.set_visible(False)
pass
pass
axs[2, 1].set_title('Outcome')
axs[2, 1].legend()
axs[2, 1].xaxis.label.set_visible(False)
axs[2, 2].remove()
pass
diabetes['BloodPressure'] = diabetes['BloodPressure'].replace(0, random.choice(diabetes[diabetes.BloodPressure != 0]['BloodPressure']))
pass
pass
pass
axs[0, 0].set_title('Pregnancies')
axs[0, 0].legend()
axs[0, 0].xaxis.label.set_visible(False)
pass
pass
axs[0, 1].set_title('Glucose')
axs[0, 1].legend()
axs[0, 1].xaxis.label.set_visible(False)
pass
pass
axs[0, 2].set_title('BloodPressure')
axs[0, 2].legend()
axs[0, 2].xaxis.label.set_visible(False)
pass
pass
axs[1, 0].set_title('Insulin')
axs[1, 0].legend()
axs[1, 0].xaxis.label.set_visible(False)
pass
pass
axs[1, 1].set_title('BMI')
axs[1, 1].legend()
axs[1, 1].xaxis.label.set_visible(False)
pass
pass
axs[1, 2].set_title('DiabetesPedigreeFunction')
axs[1, 2].legend()
axs[1, 2].xaxis.label.set_visible(False)
pass
pass
axs[2, 0].set_title('Age')
axs[2, 0].legend()
axs[2, 0].xaxis.label.set_visible(False)
pass
pass
axs[2, 1].set_title('Outcome')
axs[2, 1].legend()
axs[2, 1].xaxis.label.set_visible(False)
axs[2, 2].remove()
pass
diabetes['SkinThickness'] = diabetes['SkinThickness'].replace(0, random.choice(diabetes[diabetes.SkinThickness != 0]['SkinThickness']))
pass
pass
pass
axs[0, 0].set_title('Pregnancies')
axs[0, 0].legend()
axs[0, 0].xaxis.label.set_visible(False)
pass
pass
axs[0, 1].set_title('Glucose')
axs[0, 1].legend()
axs[0, 1].xaxis.label.set_visible(False)
pass
pass
axs[0, 2].set_title('BloodPressure')
axs[0, 2].legend()
axs[0, 2].xaxis.label.set_visible(False)
pass
pass
axs[1, 0].set_title('SkinThickness')
axs[1, 0].legend()
axs[1, 0].xaxis.label.set_visible(False)
pass
pass
axs[1, 1].set_title('BMI')
axs[1, 1].legend()
axs[1, 1].xaxis.label.set_visible(False)
pass
pass
axs[1, 2].set_title('DiabetesPedigreeFunction')
axs[1, 2].legend()
axs[1, 2].xaxis.label.set_visible(False)
pass
pass
axs[2, 0].set_title('Age')
axs[2, 0].legend()
axs[2, 0].xaxis.label.set_visible(False)
pass
pass
axs[2, 1].set_title('Outcome')
axs[2, 1].legend()
axs[2, 1].xaxis.label.set_visible(False)
axs[2, 2].remove()
pass
diabetes['Insulin'] = diabetes['Insulin'].replace(0, random.choice(diabetes[diabetes.SkinThickness != 0]['Insulin']))
diabetes['skin_missing'] = np.where(diabetes['SkinThickness'] == 0, 0, 1)
diabetes['insulin_missing'] = np.where(diabetes['Insulin'] == 0, 0, 1)
diabetes['BloodPressure_missing'] = np.where(diabetes['BloodPressure'] == 0, 0, 1)
table = pd.crosstab(diabetes['skin_missing'], diabetes['insulin_missing'], margins=True)
table1 = pd.crosstab(diabetes['BloodPressure_missing'], diabetes['insulin_missing'], margins=True)
(stat, p, dof, expected) = stats.chi2_contingency(table)
alpha = 0.05
if p < alpha:
    print('Reject H0, the missingness of SkinThickness column and Insulin column are dependent')
else:
    print('Fail to Reject H0, independent')
(stat, p, dof, expected) = stats.chi2_contingency(table1)
alpha = 0.05
if p < alpha:
    print('Reject H0, the missingness of BloodPressure column and Insulin column are dependent')
else:
    print('Fail to Reject H0, independent')
pass
fig.suptitle('Distribution of Pregnancies column')
df = pd.DataFrame(diabetes.groupby('Pregnancies')['Outcome'].mean())
df.plot(kind='bar', legend=None, ax=axes[0])
pass
df1 = pd.DataFrame(diabetes.groupby('Pregnancies')['Outcome'].mean())
df1.loc[8] = df1[df1.index >= 8].Outcome.mean()
df1.drop(index=[9, 10, 11, 12, 13, 14, 15, 17], inplace=True)
df1.plot(kind='bar', legend=None, ax=axes[1])
pass
cols = ['Before censoring', 'After censoring']
rows = ['Frequency']
for (ax, col) in zip(axes, cols):
    ax.set_title(col)
for (ax, row) in zip(axes, rows):
    pass
    pass
X = diabetes[['Pregnancies']]
X = sm.add_constant(X)
y = diabetes['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=0)