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
(fig, axes) = plt.subplots(1, 2, figsize=(12, 8))
fig.suptitle('Distribution of Pregnancies column')
df = pd.DataFrame(diabetes['Pregnancies'].value_counts())
df.plot(kind='bar', legend=None, ax=axes[0])
plt.xticks(rotation=45)
df1 = pd.DataFrame(diabetes['Pregnancies'].value_counts())
df1.loc[8] = df1[df1.index >= 8].Pregnancies.sum()
df1.drop(index=[9, 10, 11, 12, 13, 14, 15, 17], inplace=True)
df1.plot(kind='bar', legend=None, ax=axes[1])
plt.xticks(rotation=45)
cols = ['Before censoring', 'After censoring']
rows = ['Frequency']
for (ax, col) in zip(axes, cols):
    ax.set_title(col)
for (ax, row) in zip(axes, rows):
    ax.set_ylabel(row, rotation=0, size='large')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
sns.kdeplot(diabetes['Glucose'])
plt.title('Density Plot of Glucose column')
zero_glucose = diabetes[diabetes['Glucose'] == 0].shape[0]
zero_glucose_outcome = diabetes[diabetes['Glucose'] == 0].Outcome.mean()
nonzero_glucose_outcome = round(diabetes[diabetes['Glucose'] != 0].Outcome.mean(), 2)
print(f'There are {zero_glucose} records with glucose of zero. \nThe rows where glucose are zero have a mean outcome of {zero_glucose_outcome}\nThe rows where glucose are non-zero have a mean outcome of {nonzero_glucose_outcome}')
i = diabetes[diabetes['Glucose'] == 0].index
diabetes.drop(i, inplace=True)
sns.kdeplot(diabetes['BloodPressure'])
plt.title('Density plot of BloodPressure column')
sns.histplot(diabetes['SkinThickness'])
plt.ylabel('frequency')
plt.title('Histogram of SkinThickness column')
sns.histplot(diabetes[diabetes['SkinThickness'] != 0].SkinThickness)
plt.ylabel('frequency')
plt.title('Histogram of SkinThickness column without zero record')
(k2, p) = stats.normaltest(diabetes[diabetes['SkinThickness'] != 0].SkinThickness)
alpha = 0.05
print('p = {:g}'.format(p))
if p < alpha:
    print('Reject null')
else:
    print('Fail to reject null')
sns.histplot(diabetes['Insulin'])
plt.ylabel('frequency')
plt.title('Histogram of Insulin column')
zero_insulin = diabetes[diabetes['Insulin'] == 0].shape[0]
zero_insulin_percentage = round(zero_insulin / diabetes.shape[0], 2) * 100
print(f'{zero_insulin_percentage}% of the records with insulin of zero.')
sns.kdeplot(diabetes['BMI'])
plt.title('Density of BMI column')
i = diabetes[diabetes['BMI'] == 0].index
diabetes.drop(i, inplace=True)
sns.histplot(diabetes['DiabetesPedigreeFunction'])
plt.ylabel('frequency')
plt.title('Histogram of DiabetesPedigreeFunction column')
diabetes['DiabetesPedigreeFunction'] = np.where(diabetes['DiabetesPedigreeFunction'] > 1.0, 1.0, diabetes['DiabetesPedigreeFunction'])
sns.histplot(diabetes['DiabetesPedigreeFunction'])
plt.ylabel('frequency')
plt.title('Histogram of DiabetesPedigreeFunction column without zero')
sns.displot(diabetes['Age'])
plt.ylabel('frequency')
plt.title('Histogram of Age column')
no_bp = diabetes.BloodPressure == 0
no_skin = diabetes.SkinThickness == 0
no_insulin = diabetes.Insulin == 0
(fig, axs) = plt.subplots(3, 3, figsize=(15, 9))
sns.kdeplot(diabetes[no_bp].Pregnancies, label='No Blood Pressure', ax=axs[0, 0])
sns.kdeplot(diabetes[~no_bp].Pregnancies, label='Blood Pressure', ax=axs[0, 0])
axs[0, 0].set_title('Pregnancies')
axs[0, 0].legend()
axs[0, 0].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_bp].Glucose, label='No Blood Pressure', ax=axs[0, 1])
sns.kdeplot(diabetes[~no_bp].Glucose, label='Blood Pressure', ax=axs[0, 1])
axs[0, 1].set_title('Glucose')
axs[0, 1].legend()
axs[0, 1].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_bp & ~no_skin].SkinThickness, label='No Blood Pressure', ax=axs[0, 2])
sns.kdeplot(diabetes[~no_bp & ~no_skin].SkinThickness, label='Blood Pressure', ax=axs[0, 2])
axs[0, 2].set_title('SkinThickness')
axs[0, 2].legend()
axs[0, 2].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_bp & ~no_insulin].Insulin, label='No Blood Pressure', ax=axs[1, 0])
sns.kdeplot(diabetes[~no_bp & ~no_insulin].Insulin, label='Blood Pressure', ax=axs[1, 0])
axs[1, 0].set_title('Insulin')
axs[1, 0].legend()
axs[1, 0].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_bp].BMI, label='No Blood Pressure', ax=axs[1, 1])
sns.kdeplot(diabetes[~no_bp].BMI, label='Blood Pressure', ax=axs[1, 1])
axs[1, 1].set_title('BMI')
axs[1, 1].legend()
axs[1, 1].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_bp].DiabetesPedigreeFunction, label='No Skin Thickness', ax=axs[1, 2])
sns.kdeplot(diabetes[~no_bp].DiabetesPedigreeFunction, label='Skin Thickness', ax=axs[1, 2])
axs[1, 2].set_title('DiabetesPedigreeFunction')
axs[1, 2].legend()
axs[1, 2].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_bp].Age, label='No Blood Pressure', ax=axs[2, 0])
sns.kdeplot(diabetes[~no_bp].Age, label='No Blood Pressure', ax=axs[2, 0])
axs[2, 0].set_title('Age')
axs[2, 0].legend()
axs[2, 0].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_bp].Outcome, label='No Blood Pressure', ax=axs[2, 1])
sns.kdeplot(diabetes[~no_bp].Outcome, label='Blood Pressure', ax=axs[2, 1])
axs[2, 1].set_title('Outcome')
axs[2, 1].legend()
axs[2, 1].xaxis.label.set_visible(False)
axs[2, 2].remove()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
diabetes['BloodPressure'] = diabetes['BloodPressure'].replace(0, random.choice(diabetes[diabetes.BloodPressure != 0]['BloodPressure']))
(fig, axs) = plt.subplots(3, 3, figsize=(15, 9))
sns.kdeplot(diabetes[no_skin].Pregnancies, label='No Skin Thickness', ax=axs[0, 0])
sns.kdeplot(diabetes[~no_skin].Pregnancies, label='Skin Thickness', ax=axs[0, 0])
axs[0, 0].set_title('Pregnancies')
axs[0, 0].legend()
axs[0, 0].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_skin].Glucose, label='No Skin Thickness', ax=axs[0, 1])
sns.kdeplot(diabetes[~no_skin].Glucose, label='Skin Thickness', ax=axs[0, 1])
axs[0, 1].set_title('Glucose')
axs[0, 1].legend()
axs[0, 1].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_skin & ~no_bp].BloodPressure, label='No Skin Thickness', ax=axs[0, 2])
sns.kdeplot(diabetes[~no_skin & ~no_bp].BloodPressure, label='Skin Thickness', ax=axs[0, 2])
axs[0, 2].set_title('BloodPressure')
axs[0, 2].legend()
axs[0, 2].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_skin & ~no_insulin].Insulin, label='No Skin Thickness', ax=axs[1, 0])
sns.kdeplot(diabetes[~no_skin & ~no_insulin].Insulin, label='Skin Thickness', ax=axs[1, 0])
axs[1, 0].set_title('Insulin')
axs[1, 0].legend()
axs[1, 0].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_skin].BMI, label='No Skin Thickness', ax=axs[1, 1])
sns.kdeplot(diabetes[~no_skin].BMI, label='Skin Thickness', ax=axs[1, 1])
axs[1, 1].set_title('BMI')
axs[1, 1].legend()
axs[1, 1].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_skin].DiabetesPedigreeFunction, label='No Skin Thickness', ax=axs[1, 2])
sns.kdeplot(diabetes[~no_skin].DiabetesPedigreeFunction, label='Skin Thickness', ax=axs[1, 2])
axs[1, 2].set_title('DiabetesPedigreeFunction')
axs[1, 2].legend()
axs[1, 2].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_skin].Age, label='No Skin Thickness', ax=axs[2, 0])
sns.kdeplot(diabetes[~no_skin].Age, label='No Skin Thickness', ax=axs[2, 0])
axs[2, 0].set_title('Age')
axs[2, 0].legend()
axs[2, 0].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_skin].Outcome, label='No Skin Thickness', ax=axs[2, 1])
sns.kdeplot(diabetes[~no_skin].Outcome, label='Skin Thickness', ax=axs[2, 1])
axs[2, 1].set_title('Outcome')
axs[2, 1].legend()
axs[2, 1].xaxis.label.set_visible(False)
axs[2, 2].remove()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
diabetes['SkinThickness'] = diabetes['SkinThickness'].replace(0, random.choice(diabetes[diabetes.SkinThickness != 0]['SkinThickness']))
(fig, axs) = plt.subplots(3, 3, figsize=(15, 9))
sns.kdeplot(diabetes[no_insulin].Pregnancies, label='No Insulin', ax=axs[0, 0])
sns.kdeplot(diabetes[~no_insulin].Pregnancies, label='Insulin', ax=axs[0, 0])
axs[0, 0].set_title('Pregnancies')
axs[0, 0].legend()
axs[0, 0].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_insulin].Glucose, label='No Insulin', ax=axs[0, 1])
sns.kdeplot(diabetes[~no_insulin].Glucose, label='Insulin', ax=axs[0, 1])
axs[0, 1].set_title('Glucose')
axs[0, 1].legend()
axs[0, 1].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_insulin & ~no_bp].BloodPressure, label='No Insulin', ax=axs[0, 2])
sns.kdeplot(diabetes[~no_insulin & ~no_bp].BloodPressure, label='Insulin', ax=axs[0, 2])
axs[0, 2].set_title('BloodPressure')
axs[0, 2].legend()
axs[0, 2].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[~no_skin & no_insulin].SkinThickness, label='No Insulin', ax=axs[1, 0])
sns.kdeplot(diabetes[~no_skin & ~no_insulin].SkinThickness, label='Insulin', ax=axs[1, 0])
axs[1, 0].set_title('SkinThickness')
axs[1, 0].legend()
axs[1, 0].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_insulin].BMI, label='No Insulin', ax=axs[1, 1])
sns.kdeplot(diabetes[~no_insulin].BMI, label='Insulin', ax=axs[1, 1])
axs[1, 1].set_title('BMI')
axs[1, 1].legend()
axs[1, 1].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_insulin].DiabetesPedigreeFunction, label='No Insulin', ax=axs[1, 2])
sns.kdeplot(diabetes[~no_insulin].DiabetesPedigreeFunction, label='Insulin', ax=axs[1, 2])
axs[1, 2].set_title('DiabetesPedigreeFunction')
axs[1, 2].legend()
axs[1, 2].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_insulin].Age, label='No Insulin', ax=axs[2, 0])
sns.kdeplot(diabetes[~no_insulin].Age, label='Insulin', ax=axs[2, 0])
axs[2, 0].set_title('Age')
axs[2, 0].legend()
axs[2, 0].xaxis.label.set_visible(False)
sns.kdeplot(diabetes[no_insulin].Outcome, label='No Insulin', ax=axs[2, 1])
sns.kdeplot(diabetes[~no_insulin].Outcome, label='Insulin', ax=axs[2, 1])
axs[2, 1].set_title('Outcome')
axs[2, 1].legend()
axs[2, 1].xaxis.label.set_visible(False)
axs[2, 2].remove()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
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
(fig, axes) = plt.subplots(1, 2, figsize=(12, 8))
fig.suptitle('Distribution of Pregnancies column')
df = pd.DataFrame(diabetes.groupby('Pregnancies')['Outcome'].mean())
df.plot(kind='bar', legend=None, ax=axes[0])
plt.xticks(rotation=45)
df1 = pd.DataFrame(diabetes.groupby('Pregnancies')['Outcome'].mean())
df1.loc[8] = df1[df1.index >= 8].Outcome.mean()
df1.drop(index=[9, 10, 11, 12, 13, 14, 15, 17], inplace=True)
df1.plot(kind='bar', legend=None, ax=axes[1])
plt.xticks(rotation=45)
cols = ['Before censoring', 'After censoring']
rows = ['Frequency']
for (ax, col) in zip(axes, cols):
    ax.set_title(col)
for (ax, row) in zip(axes, rows):
    ax.set_ylabel(row, rotation=0, size='large')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
X = diabetes[['Pregnancies']]
X = sm.add_constant(X)
y = diabetes['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=0)