import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.classifier import StackingCVClassifier
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

def dist(i):
    pass
    pass

def dist_box(df, col):
    pass
    pass
    pass
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print('>>> Data frame shape: ', df.shape, '<<<\n')
df.head()
df.info(verbose=True)
df.describe().T
df.Outcome.value_counts()
pass
for i in range(0, len(df.columns) - 1):
    dist(i)
from pandas.plotting import scatter_matrix
p = scatter_matrix(df, figsize=(15, 15))
pass
gs = fig.add_gridspec(3, 3)
gs.update(wspace=0.5, hspace=0.25)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])
ax6 = fig.add_subplot(gs[2, 0])
ax7 = fig.add_subplot(gs[2, 1])
ax8 = fig.add_subplot(gs[2, 2])
background_color = '#c9c9ee'
color_palette = ['#f56476', '#ff8811', '#ff0040', '#ff7f6c', '#f0f66e', '#990000']
fig.patch.set_facecolor(background_color)
ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)
ax2.set_facecolor(background_color)
ax3.set_facecolor(background_color)
ax4.set_facecolor(background_color)
ax5.set_facecolor(background_color)
ax6.set_facecolor(background_color)
ax7.set_facecolor(background_color)
ax8.set_facecolor(background_color)
ax0.spines['bottom'].set_visible(False)
ax0.spines['left'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.tick_params(left=False, bottom=False)
ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.text(0.5, 0.5, 'Boxenplot plot \n features\n', horizontalalignment='center', verticalalignment='center', fontsize=18, fontweight='bold', fontfamily='serif', color='#000000')
ax1.text(-0.18, 19, 'Pregnancies', fontsize=14, fontweight='bold', fontfamily='serif', color='#000000')
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
pass
pass
pass
ax2.text(-0.1, 217, 'Glucose', fontsize=14, fontweight='bold', fontfamily='serif', color='#000000')
ax2.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
pass
pass
pass
ax3.text(-0.2, 132, 'BloodPressure', fontsize=14, fontweight='bold', fontfamily='serif', color='#000000')
ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
pass
pass
pass
ax4.text(-0.2, 110, 'SkinThickness', fontsize=14, fontweight='bold', fontfamily='serif', color='#000000')
ax4.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
pass
pass
pass
ax5.text(-0.1, 900, 'Insulin', fontsize=14, fontweight='bold', fontfamily='serif', color='#000000')
ax5.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
pass
pass
pass
ax6.text(-0.08, 77, 'BMI', fontsize=14, fontweight='bold', fontfamily='serif', color='#000000')
ax6.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
pass
pass
pass
ax7.text(-0.065, 2.8, 'DPF', fontsize=14, fontweight='bold', fontfamily='serif', color='#000000')
ax7.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
pass
pass
pass
ax8.text(-0.08, 86, 'Age', fontsize=14, fontweight='bold', fontfamily='serif', color='#000000')
ax8.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
pass
pass
pass
for s in ['top', 'right', 'left']:
    ax1.spines[s].set_visible(False)
    ax2.spines[s].set_visible(False)
    ax3.spines[s].set_visible(False)
    ax4.spines[s].set_visible(False)
    ax5.spines[s].set_visible(False)
    ax6.spines[s].set_visible(False)
    ax7.spines[s].set_visible(False)
    ax8.spines[s].set_visible(False)
pass
pass
pass
print(df.Outcome.value_counts())
p = df.Outcome.value_counts().plot(kind='bar', figsize=(10, 10))
print(df.replace(0, np.NaN).isnull().sum())
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print('\nCount of discovered nulls:\n', df.isnull().sum())
df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
df['Insulin'].fillna(np.random.choice(df['Insulin'][~df['Insulin'].isna()]), inplace=True)
df['BMI'].fillna(df['BMI'].mean(), inplace=True)
print('\nCount of nulls in the secondary date frame:\n', df.isnull().sum())
pass
for i in range(0, len(df.columns) - 1):
    dist_box(df, df.columns[i])
u = df['Insulin'].mean() + 3 * df['Insulin'].std()
l = df['Insulin'].mean() - 3 * df['Insulin'].std()
df_out_in = df[(df['Insulin'] > u) | (df['Insulin'] < l)]
print('Number of Outliers:', len(df_out_in))
df_out_in
df['Insulin'] = np.where(df['Insulin'] >= 415, df['Insulin'].mode()[0], df['Insulin'])
u = df['BloodPressure'].mean() + 3 * df['BloodPressure'].std()
l = df['BloodPressure'].mean() - 3 * df['BloodPressure'].std()
df_out_bp = df[(df['BloodPressure'] > u) | (df['BloodPressure'] < l)]
print('Number of Outliers:', len(df_out_bp))
df_out_bp
df['BloodPressure'] = np.where((df['BloodPressure'] >= 110) | (df['BloodPressure'] <= 30), df['BloodPressure'].mode()[0], df['BloodPressure'])
u = df['SkinThickness'].mean() + 3 * df['SkinThickness'].std()
l = df['SkinThickness'].mean() - 3 * df['SkinThickness'].std()
df_out_st = df[(df['SkinThickness'] > u) | (df['SkinThickness'] < l)]
print('Number of Outliers:', len(df_out_st))
df_out_st
df['SkinThickness'] = np.where(df['SkinThickness'] >= 56, df['SkinThickness'].mode()[0], df['SkinThickness'])
u = df['Pregnancies'].mean() + 3 * df['Pregnancies'].std()
l = df['Pregnancies'].mean() - 3 * df['Pregnancies'].std()
df_out_pr = df[(df['Pregnancies'] > u) | (df['Pregnancies'] < l)]
print('Number of Outliers:', len(df_out_pr))
df_out_pr
df['Pregnancies'] = np.where(df['Pregnancies'] > 13, df['Pregnancies'].mode()[0], df['Pregnancies'])
x = df.drop('Outcome', axis=1)
y = df.Outcome
(xtrain, xtest, ytrain, ytest) = train_test_split(x, y, test_size=0.3, random_state=7)
print(xtrain.shape)
print(xtest.shape)
(xtrain, xtest, ytrain, ytest) = train_test_split(x, y, random_state=7, stratify=y)
smt = SMOTE()
(xtrain, ytrain) = smt.fit_resample(xtrain, ytrain)
np.bincount(ytrain)
sc = preprocessing.StandardScaler()
xtrain = pd.DataFrame(sc.fit_transform(xtrain, ytrain), index=xtrain.index, columns=xtrain.columns)
xtrain.head()
xtest = pd.DataFrame(sc.transform(xtest), index=xtest.index, columns=xtest.columns)
xtest.head()
'sc_x = preprocessing.StandardScaler()\nx =  pd.DataFrame(sc_x.fit_transform(df.drop(["Outcome"],axis = 1),),\n        columns=[\'Pregnancies\', \'Glucose\', \'BloodPressure\', \'SkinThickness\', \'Insulin\',\n       \'BMI\', \'DiabetesPedigreeFunction\', \'Age\'])\n\nx.describe().T'
test_scores = []
train_scores = []
k_range = list(range(1, 30))
for i in k_range:
    knn = KNeighborsClassifier(i)