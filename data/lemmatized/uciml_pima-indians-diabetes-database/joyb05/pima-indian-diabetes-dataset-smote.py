import pandas as pd
import numpy as np
from numpy import nan
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.isnull().sum()
df.describe().T
pass
pass
p = df[df['Outcome'] == 1].hist(figsize=(20, 20))
pass
p = df[df['Outcome'] == 0].hist(figsize=(20, 20))
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
num_missing = (df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] == 0).sum()
print(num_missing)
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, nan)
print(df.isnull().sum())
df.isna().sum() * 100 / df.shape[0]
pass
df_pima = df
df_pima.shape
df_pima.isnull().sum()
mice = IterativeImputer(estimator=RandomForestRegressor(), random_state=0)
df_pima[['Pregnancies', 'Insulin', 'SkinThickness', 'Glucose', 'BloodPressure', 'BMI']] = mice.fit_transform(df_pima[['Pregnancies', 'Insulin', 'SkinThickness', 'Glucose', 'BloodPressure', 'BMI']])
df_pima
df_pima.isnull().sum()
pass
pass
(df_pima['Outcome'] == 1).sum() * 100 / df_pima.shape[0]
pass
pass
pass
pass
df_pima.describe()
df_pima.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(20, 15))
from sklearn.neighbors import LocalOutlierFactor
model1 = LocalOutlierFactor(n_neighbors=10)
y_pred = model1.fit_predict(df_pima)
not_outlier_index = np.where(y_pred == 1)
outlier_index = np.where(y_pred == -1)
df_pima_1 = df_pima.iloc[not_outlier_index]
df_pima
df_pima_1
df_pima_1.info()
df_pima_l = df_pima_1
df_pima_lr = df_pima_1
df_pima_xgb = df_pima_1
df_pima_q = df_pima_1
df_pima_k = df_pima_1
X = df_pima_1.drop(['Outcome'], axis=1)
y = df_pima_1.Outcome
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
(X_train, X_test, y_train, y_test) = train_test_split(X_scaled, y, random_state=10, test_size=0.3, stratify=y)
from sklearn.neighbors import KNeighborsClassifier
test_scores = []
train_scores = []
for i in range(1, 25):
    knn = KNeighborsClassifier(i)