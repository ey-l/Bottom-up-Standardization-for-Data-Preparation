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
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe()
df.info()
df.isnull().sum()
colors = ['#0A2239', '#53A2BE', '#1D84B5', '#132E32', '#176087']
pass
pass
for (idx, values) in enumerate(colors):
    pass
pass
gs = fig.add_gridspec(2, 3)
gs.update(wspace=0.5, hspace=0.25)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])
background_color = '#DFFDFF'
fig.patch.set_facecolor(background_color)
ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)
ax2.set_facecolor(background_color)
ax3.set_facecolor(background_color)
ax4.set_facecolor(background_color)
ax5.set_facecolor(background_color)
ax0.text(0.08, 550, 'Outcome', fontsize=25, color='#6D454C', weight='bold')
ax0.grid(axis='y', color='#333', linestyle=':')
ax0.tick_params(axis='both', which='major', labelsize=20)
pass
ax1.text(40, 240, 'Age', fontsize=25, color='#6D454C', weight='bold')
ax1.grid(axis='y', color='#333', linestyle=':')
ax1.tick_params(axis='both', which='major', labelsize=20)
pass
ax2.text(15, 110, 'Blood Pressure', fontsize=25, color='#6D454C', weight='bold')
ax2.grid(axis='y', color='#333', linestyle=':')
ax2.tick_params(axis='both', which='major', labelsize=20)
pass
ax3.text(100, 420, 'Insulin Levels', fontsize=25, color='#6D454C', weight='bold')
ax3.grid(axis='y', color='#333', linestyle=':')
ax3.tick_params(axis='both', which='major', labelsize=20)
pass
ax4.text(50, 111, 'Glucose Levels', fontsize=25, color='#6D454C', weight='bold')
ax4.grid(axis='y', color='#333', linestyle=':')
ax4.tick_params(axis='both', which='major', labelsize=20)
pass
ax5.text(25, 103, 'BMI', fontsize=25, color='#6D454C', weight='bold')
ax5.grid(axis='y', color='#333', linestyle=':')
ax5.tick_params(axis='both', which='major', labelsize=20)
pass
fig.suptitle('Distribution plots of different attributes', fontsize='28', weight='bold', color='#176087')
for s in ['top', 'right', 'left']:
    ax0.spines[s].set_visible(False)
    ax1.spines[s].set_visible(False)
    ax2.spines[s].set_visible(False)
    ax3.spines[s].set_visible(False)
    ax4.spines[s].set_visible(False)
    ax5.spines[s].set_visible(False)
pass
matrix = np.triu(df.corr())
pass
pass
gs = fig.add_gridspec(3, 2)
gs.update(wspace=0.5, hspace=0.25)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])
axE = fig.add_subplot(gs[2, 0])
axF = fig.add_subplot(gs[2, 1])
background_color = '#DFFDFF'
fig.patch.set_facecolor(background_color)
axA.set_facecolor(background_color)
axB.set_facecolor(background_color)
axC.set_facecolor(background_color)
axD.set_facecolor(background_color)
axE.set_facecolor(background_color)
axF.set_facecolor(background_color)
axA.tick_params(axis='both', left=False, bottom=False)
axA.set_xticklabels([])
axA.set_yticklabels([])
axA.text(0.6, 0.4, 'Age vs Pregnancies\n____________', horizontalalignment='center', verticalalignment='center', fontsize=24, fontweight='bold', fontfamily='sans-serif', color='#437F97')
axC.tick_params(axis='both', left=False, bottom=False)
axC.set_xticklabels([])
axC.set_yticklabels([])
axC.text(0.6, 0.4, 'Skin Thickness vs Insulin\n____________', horizontalalignment='center', verticalalignment='center', fontsize=24, fontweight='bold', fontfamily='sans-serif', color='#437F97')
axE.tick_params(axis='both', left=False, bottom=False)
axE.set_xticklabels([])
axE.set_yticklabels([])
axE.text(0.6, 0.4, 'Skin Thickness vs BMI\n____________', horizontalalignment='center', verticalalignment='center', fontsize=24, fontweight='bold', fontfamily='sans-serif', color='#437F97')
axB.grid(axis='y', color='#333', linestyle=':')
axB.tick_params(axis='both', which='major', labelsize=12)
pass
pass
pass
axD.grid(axis='y', color='#333', linestyle=':')
axD.tick_params(axis='both', which='major', labelsize=12)
pass
pass
pass
axF.grid(axis='y', color='#333', linestyle=':')
axF.tick_params(axis='both', which='major', labelsize=12)
pass
pass
pass
fig.suptitle('Positive Correlation', fontsize='28', weight='bold', color='#176087')
for s in ['top', 'right', 'left', 'bottom']:
    axA.spines[s].set_visible(False)
    axC.spines[s].set_visible(False)
    axE.spines[s].set_visible(False)
for s in ['top', 'right', 'left']:
    axB.spines[s].set_visible(False)
    axD.spines[s].set_visible(False)
    axF.spines[s].set_visible(False)
pass
gs = fig.add_gridspec(2, 2)
gs.update(wspace=0.5, hspace=0.25)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])
background_color = '#DFFDFF'
fig.patch.set_facecolor(background_color)
axA.set_facecolor(background_color)
axB.set_facecolor(background_color)
axC.set_facecolor(background_color)
axD.set_facecolor(background_color)
axA.tick_params(axis='both', left=False, bottom=False)
axA.set_xticklabels([])
axA.set_yticklabels([])
axA.text(0.6, 0.4, 'Age vs Skin Thickness\n____________', horizontalalignment='center', verticalalignment='center', fontsize=24, fontweight='bold', fontfamily='sans-serif', color='#437F97')
axC.tick_params(axis='both', left=False, bottom=False)
axC.set_xticklabels([])
axC.set_yticklabels([])
axC.text(0.6, 0.4, 'Age vs Insulin\n____________', horizontalalignment='center', verticalalignment='center', fontsize=24, fontweight='bold', fontfamily='sans-serif', color='#437F97')
axB.grid(axis='y', color='#333', linestyle=':')
axB.tick_params(axis='both', which='major', labelsize=12)
pass
pass
pass
axD.grid(axis='y', color='#333', linestyle=':')
axD.tick_params(axis='both', which='major', labelsize=12)
pass
pass
pass
fig.suptitle('Negative Correlation', fontsize='28', weight='bold', color='#176087')
for s in ['top', 'right', 'left', 'bottom']:
    axA.spines[s].set_visible(False)
    axC.spines[s].set_visible(False)
for s in ['top', 'right', 'left']:
    axB.spines[s].set_visible(False)
    axD.spines[s].set_visible(False)
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

def median_target(data, var):
    temp = data[data[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

def replace_median(data, columns):
    for i in columns:
        f = median_target(data, i)
        data.loc[(data['Outcome'] == 0) & data[i].isnull(), i] = f[[i]].values[0][0]
        data.loc[(data['Outcome'] == 1) & data[i].isnull(), i] = f[[i]].values[1][0]
null_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
replace_median(df, null_cols)
df.isnull().sum()
df.info()
df['Age'] = pd.qcut(df['Age'], 10, duplicates='drop')
df['BMI'] = pd.qcut(df['BMI'], 5, duplicates='drop')
df = pd.get_dummies(df)
df.head()
from sklearn.model_selection import train_test_split
X = df.drop('Outcome', axis=1)
y = df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=42)
(X_train, y_train) = os.fit_resample(X_train, y_train.ravel())
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
logmodel = LogisticRegression(max_iter=200)