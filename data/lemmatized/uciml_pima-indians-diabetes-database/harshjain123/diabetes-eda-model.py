import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import missingno as msno
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.shape
df.describe()
features = df.columns
cols = (df[features] == 0).sum()
print(cols)
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df.isnull().sum()
msno.matrix(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])
df['Glucose'].fillna(df['Glucose'].median(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)
by_Glucose_Age_Insulin_Grp = df.groupby(['Glucose'])

def fill_Insulin(series):
    return series.fillna(series.median())
df['Insulin'] = by_Glucose_Age_Insulin_Grp['Insulin'].transform(fill_Insulin)
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
by_BMI_Insulin = df.groupby(['BMI'])

def fill_Skinthickness(series):
    return series.fillna(series.mean())
df['SkinThickness'] = by_BMI_Insulin['SkinThickness'].transform(fill_Skinthickness)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace=True)
df.isnull().sum()
import matplotlib.style as style
style.available
style.use('seaborn-pastel')
labels = ['Healthy', 'Diabetic']
df['Outcome'].value_counts().plot(kind='pie', labels=labels, subplots=True, autopct='%1.0f%%', labeldistance=1.2, figsize=(9, 9))
from matplotlib.pyplot import figure, show
figure(figsize=(8, 6))
pass
ax.set_xticklabels(['Healthy', 'Diabetic'])
(healthy, diabetics) = df['Outcome'].value_counts().values
print('Samples of diabetic people: ', diabetics)
print('Samples of healthy people: ', healthy)
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
ax0.text(0.5, 0.5, 'Boxenplot plot for various\n features\n_________________\n\n CREDIT: Naman Manchanda', horizontalalignment='center', verticalalignment='center', fontsize=18, fontweight='bold', fontfamily='serif', color='#000000')
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
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
pass
pass
pass
pass
pass
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
print('Number transactions x_train dataset: ', x_train.shape)
print('Number transactions y_train dataset: ', y_train.shape)
print('Number transactions x_test dataset: ', x_test.shape)
print('Number transactions y_test dataset: ', y_test.shape)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, accuracy_score, auc
from sklearn.svm import SVC
model = SVC(kernel='rbf')