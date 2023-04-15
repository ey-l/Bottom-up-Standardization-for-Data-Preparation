import pandas as pd
import numpy as np
import scipy as sp
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SelectFromModel
import statsmodels.api as sm
import statsmodels.stats.api as sms
import seaborn as sns
sns.set()
sns.set_style('whitegrid')
import matplotlib.pyplot as plt

pima_diabetes_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
pima_diabetes_data.info()
y = pima_diabetes_data.loc[:, ['Outcome']]
X = pima_diabetes_data
X = X.drop(['Outcome'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=2021, stratify=y)
X_train.info()
y_train.info()
print(X_train.isnull().sum())
print(y_train.isnull().sum())
X_train.describe().round(1)
y_train_cat = y_train.astype('category')
y_train_cat.describe()
print(X_train.columns)
plt.hist(X_train['Pregnancies'], color='green')
plt.xlabel('Number of Times Pregnant')
plt.ylabel('Patient Count')
plt.title("Distribution of 'Pregnancies' Predictor Variable")
plt.hist(X_train['Glucose'], color='green')
plt.xlabel('Plasma glucose concentration after tolerance test')
plt.ylabel('Patient Count')
plt.title("Distribution of 'Glucose' Predictor Variable")
plt.hist(X_train['BloodPressure'], color='green')
plt.xlabel('Diastolic blood pressure (mm Hg)')
plt.ylabel('Patient Count')
plt.title("Distribution of 'BloodPressure' Predictor Variable")
plt.hist(X_train['SkinThickness'], color='green')
plt.xlabel('Triceps skin fold thickness (mm)')
plt.ylabel('Patient Count')
plt.title("Distribution of 'SkinThickness' Predictor Variable")
plt.hist(X_train['Insulin'], color='green')
plt.xlabel('2-Hour serum insulin (mu U/ml)')
plt.ylabel('Patient Count')
plt.title("Distribution of 'Insulin' Predictor Variable")
plt.hist(X_train['BMI'], color='green')
plt.xlabel('Body mass index (weight in kg/(height in m)^2)')
plt.ylabel('Patient Count')
plt.title("Distribution of 'BMI' Predictor Variable")
plt.hist(X_train['DiabetesPedigreeFunction'], color='green')
plt.xlabel('Diabetes pedigree function')
plt.ylabel('Patient Count')
plt.title("Distribution of 'Diabetespedigreefunction' Predictor Variable")
plt.hist(X_train['Age'], color='green')
plt.xlabel('Age(years)')
plt.ylabel('Patient Count')
plt.title("Distribution of 'Age' Predictor Variable")
combined_training_data = pd.concat([X_train, y_train], axis=1)
combined_training_data.head()
(f, axes) = plt.subplots(4, 2, figsize=(20, 15))
sns.boxplot(x='Outcome', y='Pregnancies', data=combined_training_data, orient='v', ax=axes[0, 0])
sns.boxplot(x='Outcome', y='Glucose', data=combined_training_data, orient='v', ax=axes[0, 1])
sns.boxplot(x='Outcome', y='BloodPressure', data=combined_training_data, orient='v', ax=axes[1, 0])
sns.boxplot(x='Outcome', y='SkinThickness', data=combined_training_data, orient='v', ax=axes[1, 1])
sns.boxplot(x='Outcome', y='Insulin', data=combined_training_data, orient='v', ax=axes[2, 0])
sns.boxplot(x='Outcome', y='BMI', data=combined_training_data, orient='v', ax=axes[2, 1])
sns.boxplot(x='Outcome', y='DiabetesPedigreeFunction', data=combined_training_data, orient='v', ax=axes[3, 0])
sns.boxplot(x='Outcome', y='Age', data=combined_training_data, orient='v', ax=axes[3, 1])
corr_combined = combined_training_data
act_corr = corr_combined.corr()
matrix = np.tril(act_corr)
(f, ax) = plt.subplots(figsize=(15, 12))
sns.heatmap(act_corr, vmax=0.8, annot=True, mask=matrix)
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=2021)
opt_feat_num_rfecv = RFECV(estimator=rf_classifier, step=1, cv=StratifiedKFold(3), scoring='balanced_accuracy', min_features_to_select=1)