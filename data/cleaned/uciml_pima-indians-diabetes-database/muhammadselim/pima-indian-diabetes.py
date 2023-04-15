from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')

from termcolor import colored
warnings.filterwarnings('ignore')
diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes.head(10)
diabetes.info()
diabetes.describe()
diabetes.isna().sum()
diabetes[diabetes.duplicated()].count()

cols = list(diabetes.drop('Outcome', axis=1).columns)
target = ['Outcome']
print(f"The columns are : {colored(cols, 'green')}")
print(f"The target is   : {colored(target, 'green')}")
diabetes.describe().transpose()
plt.figure(figsize=(6, 8))
sns.set_style(style='white')
sns.countplot(diabetes['Outcome'])
fig = plt.figure(figsize=(18, 7))
gs = fig.add_gridspec(1, 2)
gs.update(wspace=0.3, hspace=0.15)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
background_color = '#c9c9ee'
color_palette = ['#f56476', '#ff8811', '#001427', '#6369d1', '#f0f66e']
fig.patch.set_facecolor(background_color)
ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)
ax0.text(0.5, 0.5, 'Count of the target\n___________', horizontalalignment='center', verticalalignment='center', fontsize=18, fontweight='bold', fontfamily='serif', color='#000000')
ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.tick_params(left=False, bottom=False)
ax1.text(0.45, 510, 'Output', fontsize=14, fontweight='bold', fontfamily='serif', color='#000000')
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
sns.countplot(ax=ax1, data=diabetes, x='Outcome', palette=color_palette)
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_xticklabels(['Low chances of diabetes(0)', 'High chances of diabetes(1)'])
ax0.spines['top'].set_visible(False)
ax0.spines['left'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
sns.color_palette('hls', 8)
sns.pairplot(diabetes, hue='Outcome')
plt.figure(figsize=(20, 20))
sns.light_palette('seagreen', as_cmap=True)
sns.heatmap(diabetes.corr(), annot=True)
plt.figure(figsize=(15, 15))
sns.light_palette('seagreen', as_cmap=True)
sns.heatmap(diabetes.corr(), cmap='viridis', annot=True)
import missingno as msno
p = msno.bar(diabetes)
p = sns.pairplot(diabetes, hue='Outcome')
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(diabetes.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
y = diabetes.Outcome
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=1 / 3, random_state=42, stratify=y)
from sklearn.neighbors import KNeighborsClassifier
test_scores = []
train_scores = []
for i in range(1, 15):
    knn = KNeighborsClassifier(i)