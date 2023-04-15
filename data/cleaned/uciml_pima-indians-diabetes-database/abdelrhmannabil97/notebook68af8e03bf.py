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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
sns.pairplot(df, hue='Outcome')
df_c = df.iloc[:, :]
df_c.iloc[:, 1:-1] = df_c.iloc[:, 1:-1].replace(0, np.NaN)
df_c.isnull().sum()
df_c.hist(figsize=(20, 20))
df_c.Glucose.fillna(df_c.Glucose.mean(), inplace=True)
df_c.BloodPressure.fillna(df_c.BloodPressure.mean(), inplace=True)
df_c.SkinThickness.fillna(df_c.SkinThickness.median(), inplace=True)
df_c.Insulin.fillna(df_c.Insulin.median(), inplace=True)
df_c.BMI.fillna(df_c.BMI.median(), inplace=True)
df_c.hist(figsize=(20, 20))
(fig, ax) = plt.subplots(figsize=(15, 8))
df_copy = df_c.iloc[:, :-1]
target = df_c.iloc[:, -1].to_frame()
sns.boxplot(data=df_copy)
df_copy
(X_train, X_test, y_train, y_test) = train_test_split(df_copy, target, test_size=0.2, shuffle=True, random_state=0)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
X_train.shape
sm = SMOTE()
(X_res, y_res) = sm.fit_resample(X_train, y_train)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_res)
X_test = scaler.transform(X_test)
print('X train before Normalization')
print(X_train[0:5])
print('\nX train after Normalization')
print(X_train[0:5])
df2 = pd.DataFrame(data=np.c_[X_train, y_res], columns=df.columns)
df2
sns.pairplot(df2, hue='Outcome')
knn = KNeighborsClassifier(20)