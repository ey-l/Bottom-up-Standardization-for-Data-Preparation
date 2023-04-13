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
pass
pass
pass
pass
pass
pass
pass
print('Median of Pregnancies: ', df['Pregnancies'].median())
print('Maximum of Pregnancies: ', df['Pregnancies'].max())
df['Pregnancies'].value_counts()
pass
pass
df['Outcome'].value_counts().plot.pie(explode=[0.1, 0], autopct='%1.1f%%', labels=['No', 'Yes'], shadow=True, ax=ax[1])
pass
pass
pass
pass
pass
print('Median of Glucose: ', df['Glucose'].median())
print('Maximum of Glucose: ', df['Glucose'].max())
print('Mean of Glucose: ', df['Glucose'].mean())
print('Rows with Glucose value of 0: ', df[df['Glucose'] == 0].shape[0])
pass
pass
pass
pass
pass
print('Median of Blood Pressure: ', df['BloodPressure'].median())
print('Maximum of Blood Pressure: ', df['BloodPressure'].max())
print('Mean of Pressure: ', df['BloodPressure'].mean())
print('Rows with BloodPressure value of 0: ', df[df['BloodPressure'] == 0].shape[0])
pass
pass
pass
pass
pass
print('Rows with Insulin value of 0: ', df[df['Insulin'] == 0].shape[0])
pass
pass
pass
pass
pass
print('Median of BMI: ', df['BMI'].median())
print('Maximum of BMI: ', df['BMI'].max())
print('Mean of BMI: ', df['BMI'].mean())
print('Rows with BMI value of 0: ', df[df['BMI'] == 0].shape[0])
pass
pass
pass
pass
pass
print('Median of DiabetesPedigreeFunction: ', df['DiabetesPedigreeFunction'].median())
print('Maximum of DiabetesPedigreeFunction: ', df['DiabetesPedigreeFunction'].max())
print('Mean of DiabetesPedigreeFunction: ', df['DiabetesPedigreeFunction'].mean())
pass
pass
pass
pass
pass
print('Median of Age: ', df['Age'].median())
print('Maximum of Age: ', df['Age'].max())
print('Mean of Age: ', df['Age'].mean())
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
corr_matrix = df.corr()
pass
pass
corr_matrix['Outcome'].sort_values(ascending=False)
newdf = df
newdf['Glucose_cat'] = pd.cut(newdf['Glucose'], bins=[-1, 40, 80, 120, 160, np.inf], labels=[1, 2, 3, 4, 5])
newdf['Glucose_cat'].value_counts()
pass
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