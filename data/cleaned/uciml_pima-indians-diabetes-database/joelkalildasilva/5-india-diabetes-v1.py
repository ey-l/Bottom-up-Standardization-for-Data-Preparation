
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msgn
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head(15)
df.Outcome.value_counts()
df.columns
sns.pairplot(df, hue='Outcome')
for col in df.columns:
    plt.figure(figsize=[10, 5])
    sns.distplot(df[col])
df_ = df.replace(0, np.nan)
df_.Pregnancies = df.Pregnancies
df_.Outcome = df.Outcome
df_.head(15)
df_.count()
msgn.bar(df_)
df_.info()
values = {'Glucose': np.random.normal(df_.Glucose.mean(), df_.Glucose.std()), 'BloodPressure': np.random.normal(df_.BloodPressure.mean(), df_.BloodPressure.std()), 'SkinThickness': np.random.normal(df_.SkinThickness.mean(), df_.SkinThickness.std()), 'BMI': np.random.normal(df_.BMI.mean(), df_.BMI.std())}
values
df_ = df_.fillna(value=values)
df_Insulin_nan = df_[np.isnan(df_.Insulin)]
df_Insulin = df_.drop(df_Insulin_nan.index)
X = df_Insulin.drop(['Insulin'], axis=1)
y = df_Insulin.Insulin
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1, random_state=1)
from sklearn import ensemble
rf = ensemble.RandomForestRegressor()