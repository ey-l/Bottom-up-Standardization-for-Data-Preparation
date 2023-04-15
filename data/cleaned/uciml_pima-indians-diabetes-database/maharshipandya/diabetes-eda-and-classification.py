import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df.info()
df.describe()

def count_na(df, col):
    print(f'Null values in {col}: ', df[col].isna().sum())
for feat in df.columns:
    count_na(df, feat)
sns.set_style('darkgrid')
sns.set_palette('viridis')
(fig1, ax1) = plt.subplots(1, 2, figsize=(20, 7))
(fig2, ax2) = plt.subplots(figsize=(20, 7))
sns.histplot(data=df, x='Pregnancies', kde=True, ax=ax1[0])
sns.boxplot(data=df, x='Pregnancies', ax=ax1[1])
sns.violinplot(data=df, x='Pregnancies', ax=ax2)

print('Median of Pregnancies: ', df['Pregnancies'].median())
print('Maximum of Pregnancies: ', df['Pregnancies'].max())
df['Pregnancies'].value_counts()
(fig, ax) = plt.subplots(1, 2, figsize=(20, 7))
sns.countplot(data=df, x='Outcome', ax=ax[0])
df['Outcome'].value_counts().plot.pie(explode=[0.1, 0], autopct='%1.1f%%', labels=['No', 'Yes'], shadow=True, ax=ax[1])

(fig3, ax3) = plt.subplots(1, 2, figsize=(20, 7))
(fig4, ax4) = plt.subplots(figsize=(20, 7))
sns.histplot(data=df, x='Glucose', kde=True, ax=ax3[0])
sns.boxplot(data=df, x='Glucose', ax=ax3[1])
sns.violinplot(data=df, x='Glucose', ax=ax4)

print('Median of Glucose: ', df['Glucose'].median())
print('Maximum of Glucose: ', df['Glucose'].max())
print('Mean of Glucose: ', df['Glucose'].mean())
print('Rows with Glucose value of 0: ', df[df['Glucose'] == 0].shape[0])
(fig5, ax5) = plt.subplots(1, 2, figsize=(20, 7))
(fig6, ax6) = plt.subplots(figsize=(20, 7))
sns.histplot(data=df, x='BloodPressure', kde=True, ax=ax5[0])
sns.boxplot(data=df, x='BloodPressure', ax=ax5[1])
sns.violinplot(data=df, x='BloodPressure', ax=ax6)

print('Median of Blood Pressure: ', df['BloodPressure'].median())
print('Maximum of Blood Pressure: ', df['BloodPressure'].max())
print('Mean of Pressure: ', df['BloodPressure'].mean())
print('Rows with BloodPressure value of 0: ', df[df['BloodPressure'] == 0].shape[0])
(fig7, ax7) = plt.subplots(1, 2, figsize=(20, 7))
(fig8, ax8) = plt.subplots(figsize=(20, 7))
sns.histplot(data=df, x='Insulin', kde=True, ax=ax7[0])
sns.boxplot(data=df, x='Insulin', ax=ax7[1])
sns.violinplot(data=df, x='Insulin', ax=ax8)

print('Rows with Insulin value of 0: ', df[df['Insulin'] == 0].shape[0])
(fig9, ax9) = plt.subplots(1, 2, figsize=(20, 7))
(fig10, ax10) = plt.subplots(figsize=(20, 7))
sns.histplot(data=df, x='BMI', kde=True, ax=ax9[0])
sns.boxplot(data=df, x='BMI', ax=ax9[1])
sns.violinplot(data=df, x='BMI', ax=ax10)

print('Median of BMI: ', df['BMI'].median())
print('Maximum of BMI: ', df['BMI'].max())
print('Mean of BMI: ', df['BMI'].mean())
print('Rows with BMI value of 0: ', df[df['BMI'] == 0].shape[0])
(fig11, ax11) = plt.subplots(1, 2, figsize=(20, 7))
(fig12, ax12) = plt.subplots(figsize=(20, 7))
sns.histplot(data=df, x='DiabetesPedigreeFunction', kde=True, ax=ax11[0])
sns.boxplot(data=df, x='DiabetesPedigreeFunction', ax=ax11[1])
sns.violinplot(data=df, x='DiabetesPedigreeFunction', ax=ax12)

print('Median of DiabetesPedigreeFunction: ', df['DiabetesPedigreeFunction'].median())
print('Maximum of DiabetesPedigreeFunction: ', df['DiabetesPedigreeFunction'].max())
print('Mean of DiabetesPedigreeFunction: ', df['DiabetesPedigreeFunction'].mean())
(fig13, ax13) = plt.subplots(1, 2, figsize=(20, 7))
(fig14, ax14) = plt.subplots(figsize=(20, 7))
sns.histplot(data=df, x='Age', kde=True, ax=ax13[0])
sns.boxplot(data=df, x='Age', ax=ax13[1])
sns.violinplot(data=df, x='Age', ax=ax14)

print('Median of Age: ', df['Age'].median())
print('Maximum of Age: ', df['Age'].max())
print('Mean of Age: ', df['Age'].mean())
(fig15, ax15) = plt.subplots(figsize=(20, 8))
sns.histplot(data=df, x='Glucose', hue='Outcome', shrink=0.8, multiple='fill', kde=True, ax=ax15)

(fig16, ax16) = plt.subplots(figsize=(20, 8))
sns.histplot(data=df, x='BloodPressure', hue='Outcome', shrink=0.8, multiple='dodge', kde=True, ax=ax16)

(fig17, ax17) = plt.subplots(figsize=(20, 8))
sns.histplot(data=df, x='BMI', hue='Outcome', shrink=0.8, multiple='fill', kde=True, ax=ax17)

(fig18, ax18) = plt.subplots(figsize=(20, 8))
sns.histplot(data=df, x='Age', hue='Outcome', shrink=0.8, multiple='dodge', kde=True, ax=ax18)

(fig19, ax19) = plt.subplots(figsize=(20, 8))
sns.histplot(data=df, x='Pregnancies', hue='Outcome', shrink=0.8, multiple='fill', kde=True, ax=ax19)

corr_matrix = df.corr()
(fig20, ax20) = plt.subplots(figsize=(20, 7))
dataplot = sns.heatmap(data=corr_matrix, annot=True, ax=ax20)

corr_matrix['Outcome'].sort_values(ascending=False)
newdf = df
newdf['Glucose_cat'] = pd.cut(newdf['Glucose'], bins=[-1, 40, 80, 120, 160, np.inf], labels=[1, 2, 3, 4, 5])
newdf['Glucose_cat'].value_counts()
(fig21, ax21) = plt.subplots(figsize=(20, 7))
newdf['Glucose_cat'].hist(ax=ax21)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=3301)
for (train_index, test_index) in split.split(newdf, newdf['Glucose_cat']):
    strat_train_set = newdf.loc[train_index]
    strat_test_set = newdf.loc[test_index]

def get_glucose_proportions(ndf):
    print(ndf['Glucose_cat'].value_counts() / len(ndf))
print('Entire Dataset: ')
get_glucose_proportions(newdf)
print('\n')
print('-' * 30)
print('\nTesting set: ')
get_glucose_proportions(strat_test_set)
for set_ in (strat_train_set, strat_test_set):
    set_.drop(columns=['Glucose_cat'], inplace=True)
meds = []
feats = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for feat in feats:
    meds.append(strat_train_set[feat].median())
print('Medians are: ', meds)

def replace_with_median(ndf, feat, value):
    ndf[feat] = ndf[feat].replace(0, value)
for (i, feat) in enumerate(feats):
    replace_with_median(strat_train_set, feat, meds[i])
    replace_with_median(strat_test_set, feat, meds[i])
X_train = strat_train_set.drop(columns='Outcome')
y_train = strat_train_set['Outcome']
X_test = strat_test_set.drop(columns='Outcome')
y_test = strat_test_set['Outcome']
stdscaler = StandardScaler()