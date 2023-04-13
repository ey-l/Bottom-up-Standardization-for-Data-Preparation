import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv', engine='python')
df.head()
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)
print(x_train.info())
print('===============')
print(x_test.info())
print(np.sum(y_train) / len(y_train))
print(np.sum(y_test) / len(y_test))
x_train.describe()
x_test.describe()
train_columns = list(x_train.columns)[:-1]
for col in train_columns:
    q1 = x_train[col].quantile(q=0.25)
    q3 = x_train[col].quantile(q=0.75)
    iqr = q3 - q1
    condi = (x_train[col] < q1 - iqr * 1.5) | (x_train[col] > q3 + iqr * 1.5)
    outliers = len(x_train[condi])
    print('train', col, outliers)
test_columns = list(x_test.columns)[:-1]
for col in test_columns:
    q1 = x_test[col].quantile(q=0.25)
    q3 = x_test[col].quantile(q=0.75)
    iqr = q3 - q1
    condi = (x_test[col] < q1 - iqr * 1.5) | (x_test[col] > q3 + iqr * 1.5)
    outliers = len(x_test[condi])
    print('train', col, outliers)
train_columns = list(x_train.columns)[:-1]
for col in train_columns:
    q1 = x_train[col].quantile(q=0.25)
    q3 = x_train[col].quantile(q=0.75)
    iqr = q3 - q1
    condi = (x_train[col] < q1 - iqr * 1.5) | (x_train[col] > q3 + iqr * 1.5)
    outliers = x_train[condi]
    if len(outliers) > 0:
        x_train[col].loc[outliers.index] = x_train[col].median()
test_columns = list(x_test.columns)[:-1]
for col in test_columns:
    q1 = x_test[col].quantile(q=0.25)
    q3 = x_test[col].quantile(q=0.75)
    iqr = q3 - q1
    condi = (x_test[col] < q1 - iqr * 1.5) | (x_test[col] > q3 + iqr * 1.5)
    outliers = x_test[condi]
    if len(outliers) > 0:
        x_test[col].loc[outliers.index] = x_test[col].median()
x_train.describe()
x_test.describe()
std = StandardScaler()