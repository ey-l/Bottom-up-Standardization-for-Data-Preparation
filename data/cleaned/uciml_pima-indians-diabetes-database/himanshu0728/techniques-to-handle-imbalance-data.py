import numpy as np
import pandas as pd
import seaborn as sns
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
sns.countplot(data['Outcome'])
(class_count_0, class_count_1) = data['Outcome'].value_counts()
class_0 = data[data['Outcome'] == 0]
class_1 = data[data['Outcome'] == 1]
print('class 0 : ', class_0.shape)
print('class 1 : ', class_1.shape)
class_0_under = class_0.sample(class_count_1)
print('class_0_under : ', class_0_under.shape)
print('class 1 : ', class_1.shape)
class_1_over = class_1.sample(class_count_0, replace=True)
print('class 0 : ', class_0.shape)
print('class_1_over : ', class_1_over.shape)
type(class_0_under)
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=10)
X = data.drop(['Outcome'], axis=1)
y = data[['Outcome']]
(X_rus, y_rus) = rus.fit_resample(X, y)
sns.countplot(y_rus['Outcome'])
sns.countplot(y['Outcome'])
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=10)
X = data.drop(['Outcome'], axis=1)
y = data[['Outcome']]
(X_ros, y_ros) = ros.fit_resample(X, y)
sns.countplot(y_ros['Outcome'])
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=10)
X = data.drop(['Outcome'], axis=1)
y = data[['Outcome']]
(X_smote, y_smote) = smote.fit_resample(X, y)
sns.countplot(y_smote['Outcome'])
from imblearn.under_sampling import NearMiss
nm = NearMiss()
X = data.drop(['Outcome'], axis=1)
y = data[['Outcome']]
(X_nm, y_nm) = nm.fit_resample(X, y)
sns.countplot(y_nm['Outcome'])