import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
print('Number of rows present in the dataset are: ', df.shape)
df.info()
df.describe().T
import seaborn as sns
from itertools import cycle
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
sns.countplot(df['Outcome'])

df['Outcome'].value_counts()
(fig, ax) = plt.subplots()
labels = ['Diabetic', 'Non-Diabetic']
percentages = [34.89, 65.1]
explode = (0.1, 0)
ax.pie(percentages, explode=explode, labels=labels, autopct='%1.0f%%', shadow=False, startangle=0, pctdistance=1.2, labeldistance=1.4)
ax.legend(frameon=False, bbox_to_anchor=(1.5, 0.8))

for col in df.columns:
    print('The minimum value fore the columns {} is {}'.format(col, df[col].min()))

def msv_1(data, thresh=20, color='black', edgecolor='black', height=3, width=15):
    plt.figure(figsize=(width, height))
    percentage = data.isnull().mean() * 100
    percentage.sort_values(ascending=False).plot.bar(color=color, edgecolor=edgecolor)
    plt.axhline(y=thresh, color='r', linestyle='-')
    plt.title('Missing values percentage per column', fontsize=20, weight='bold')
    plt.text(len(data.isnull().sum() / len(data)) / 1.7, thresh + 2.5, f'Columns with more than {thresh}% missing values', fontsize=12, color='crimson', ha='left', va='top')
    plt.text(len(data.isnull().sum() / len(data)) / 1.7, thresh - 0.5, f'Columns with less than {thresh}% missing values', fontsize=12, color='green', ha='left', va='top')
    plt.xlabel('Columns', size=15, weight='bold')
    plt.ylabel('Missing values percentage')
    plt.yticks(weight='bold')

msv_1(df, 20, color=sns.color_palette('Reds', 15))
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
msv_1(df, 20, color=sns.color_palette('Reds', 15))
(fig, axes) = plt.subplots(4, 2, figsize=(15, 10))
axes = axes.flatten()
ax_idx = 0
columns = df.drop('Outcome', axis=1).columns
for col in columns:
    df[col].plot(kind='hist', ax=axes[ax_idx], title=col, color=next(color_cycle))
    ax_idx += 1
plt.suptitle('Sales Trend according to Departments')
plt.tight_layout()

from scipy.stats import skew
for col in df.drop('Outcome', axis=1).columns:
    print('Skewness for the column {} is {}'.format(col, df[col].skew()))
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].median())
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']:
    df[col] = df[col].fillna(df[col].mean())
msv_1(df, 10, color=sns.color_palette('Greens', 15))
df.isnull().sum()

def mean_target(var):
    """
    A function that will return the mean values for 'var' column depending on whether the person
    is diabetic or not
    """
    return pd.DataFrame(df.groupby('Outcome').mean()[var])

def distplot(col_name):
    """
    A function that will plot the distribution of column 'col_name' for diabetic and non-diabetic people separately
    """
    plt.figure()
    ax = sns.distplot(df[col_name][df.Outcome == 1], color='red', rug=True)
    sns.distplot(df[col_name][df.Outcome == 0], color='lightblue', rug=True)
    plt.legend(['Diabetes', 'No Diabetes'])
distplot('Pregnancies')
mean_target('Pregnancies')
distplot('Insulin')
mean_target('Insulin')
distplot('BloodPressure')
mean_target('BloodPressure')
distplot('Glucose')
mean_target('Glucose')
sns.boxplot(x='Outcome', y='Age', data=df)
plt.title('Age vs Outcome')

sns.boxplot(x='Outcome', y='BloodPressure', data=df, palette='Blues')
plt.title('BP vs Outcome')

sns.jointplot(x='Age', y='BloodPressure', data=df, kind='reg', color='green')
my_pal = {0: 'lightgreen', 1: 'lightblue'}
sns.boxplot(x='Outcome', y='DiabetesPedigreeFunction', data=df, palette=my_pal)
plt.title('DPF vs Outcome')

my_pal = {0: 'lightgrey', 1: 'lightyellow'}
sns.boxplot(x='Outcome', y='Glucose', data=df, palette=my_pal)
plt.title('Glucose vs Outcome')

sns.jointplot(x='Insulin', y='Glucose', data=df, kind='reg', color='red')

sns.boxplot(x='Outcome', y='Insulin', data=df)
plt.title('Insulin vs Outcome')

my_pal = {0: 'lightyellow', 1: 'lightpink'}
sns.boxplot(x='Outcome', y='BMI', data=df, palette=my_pal)
plt.title('BMI vs Outcome')

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
(f, ax) = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5}, annot=True)
from sklearn.model_selection import train_test_split
X = df.drop('Outcome', axis=1)
y = df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X_test = pd.DataFrame(sc.fit_transform(X_test), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluation(model, x_train_std, y_train, x_test, y_test, train=True):
    """
    A function that returns the score of every evaluation metrics
    """
    if train == True:
        pred = model.predict(x_train_std)
        classifier_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print('Train Result:\n================================================')
        print(f'Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%')
        print('_______________________________________________')
        print(f'F1 Score: {round(f1_score(y_train, pred), 2)}')
        print('_______________________________________________')
        print(f'CLASSIFICATION REPORT:\n{classifier_report}')
        print('_______________________________________________')
        print(f'Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n')
    if train == False:
        pred = model.predict(x_test)
        classifier_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print('Test Result:\n================================================')
        print(f'Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%')
        print('_______________________________________________')
        print(f'F1 Score: {round(f1_score(y_test, pred), 2)}')
        print('_______________________________________________')
        print(f'CLASSIFICATION REPORT:\n{classifier_report}')
        print('_______________________________________________')
        print(f'Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n')
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')