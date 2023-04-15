import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.isnull().sum()
data.info()
data.describe()

def BoxPlot(x, y, axis):
    """
    This function return Box plot of give data with respect Outcome means 
    0 : Not Diabetes 
    1 : Diabetes
    
    
    """
    return sns.boxplot(x=x, y=y, ax=axis, data=data)
(fig, ax) = plt.subplots(4, 2, figsize=(15, 15))
ax1 = ax[0, 0]
ax2 = ax[0, 1]
ax3 = ax[1, 0]
ax4 = ax[1, 1]
ax5 = ax[2, 0]
ax6 = ax[2, 1]
ax7 = ax[3, 0]
ax8 = ax[3, 1]
BoxPlot('Outcome', 'Glucose', ax1)
BoxPlot('Outcome', 'BloodPressure', ax2)
BoxPlot('Outcome', 'SkinThickness', ax3)
BoxPlot('Outcome', 'Insulin', ax4)
BoxPlot('Outcome', 'BMI', ax5)
BoxPlot('Outcome', 'DiabetesPedigreeFunction', ax6)
BoxPlot('Outcome', 'Age', ax7)
BoxPlot('Outcome', 'Pregnancies', ax8)


def Outliers(col):
    (data_mean, data_std) = (np.mean(col), np.std(col))
    cut_off = data_std * 3
    (lower, upper) = (data_mean - cut_off, data_mean + cut_off)
    outliers = [x for x in col if x < lower or x > upper]
    return len(outliers)
print('Outliers in Glucose :', Outliers(data['Glucose']))
print('Outliers in BloodPressure :', Outliers(data['BloodPressure']))
print('Outliers in SkinThickness :', Outliers(data['SkinThickness']))
print('Outliers in BMI :', Outliers(data['BMI']))
print('Outliers in DiabetesPedigreeFunction :', Outliers(data['DiabetesPedigreeFunction']))
print('Outliers in Age :', Outliers(data['Age']))
print('Outliers in Pregnancies :', Outliers(data['Pregnancies']))

def ScatterPlot(x, y, axis):
    """
    This function return Scatter plot of give data with respect Outcome means 
    0 : Not Diabetes 
    1 : Diabetes
    
    
    """
    return sns.scatterplot(x=x, y=y, hue='Outcome', ax=axis, data=data)
(fig, ax) = plt.subplots(3, 2, figsize=(15, 15))
ax1 = ax[0, 0]
ax2 = ax[0, 1]
ax3 = ax[1, 0]
ax4 = ax[1, 1]
ax5 = ax[2, 0]
ax6 = ax[2, 1]
ScatterPlot('Glucose', 'Age', ax1)
ScatterPlot('BloodPressure', 'Age', ax2)
ScatterPlot('SkinThickness', 'Age', ax3)
ScatterPlot('Insulin', 'Age', ax4)
ScatterPlot('BMI', 'Age', ax5)
ScatterPlot('DiabetesPedigreeFunction', 'Age', ax6)

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), square=True, cmap='coolwarm', annot=True)

def OutcomeAnalysis(col_name, axis):
    return data.groupby('Outcome')[col_name].mean().sort_values().plot(kind='bar', color='coral', ax=axis)
(fig, ax) = plt.subplots(3, 2, figsize=(15, 15))
ax1 = ax[0, 0]
ax2 = ax[0, 1]
ax3 = ax[1, 0]
ax4 = ax[1, 1]
ax5 = ax[2, 0]
ax6 = ax[2, 1]
OutcomeAnalysis('Glucose', ax1)
ax1.set_title('Avg.Glucose level')
OutcomeAnalysis('BloodPressure', ax2)
ax2.set_title('Avg.BloodPressure')
OutcomeAnalysis('SkinThickness', ax3)
ax3.set_title('Avg.SkinThickness')
OutcomeAnalysis('Insulin', ax4)
ax4.set_title('Avg.Insulin')
OutcomeAnalysis('BMI', ax5)
ax5.set_title('Avg.BMI')
OutcomeAnalysis('DiabetesPedigreeFunction', ax6)
ax6.set_title('Avg.DiabetesPedigreeFunction')

(fig, ax) = plt.subplots(4, 2, figsize=(15, 15))
dp = sns.distplot(data['Pregnancies'], ax=ax[0, 0])
dp = sns.distplot(data['Glucose'], ax=ax[0, 1])
dp = sns.distplot(data['BloodPressure'], ax=ax[1, 0])
dp = sns.distplot(data['SkinThickness'], ax=ax[1, 1])
dp = sns.distplot(data['Insulin'], ax=ax[2, 0])
dp = sns.distplot(data['BMI'], ax=ax[2, 1])
dp = sns.distplot(data['DiabetesPedigreeFunction'], ax=ax[3, 0])
dp = sns.distplot(data['Age'], ax=ax[3, 1])

X = data.drop(['Outcome'], axis=1)
y = data['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3)
sc = StandardScaler()
X2_train = sc.fit_transform(X_train)
X2_test = sc.fit_transform(X_test)
y2_train = y_train
y2_test = y_test

def OptimalKNN(X_train, X_test, y_train, y_test):
    max_k = 50
    f1_scores = list()
    error_rates = list()
    for k in range(1, max_k):
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')