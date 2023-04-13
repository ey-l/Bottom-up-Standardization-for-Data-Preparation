import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print('Dataset has {} rows and {} columns'.format(df.shape[0], df.shape[1]))
orig_df = df.copy(deep=True)
target = 'Outcome'
features = [i for i in df.columns.values if i not in [target]]
df.info()
unique_rows = df[features].nunique().sort_values()
num_features = []
cat_features = []
for i in range(df[features].shape[1]):
    if unique_rows.values[i] <= 7:
        cat_features.append(unique_rows.index[i])
    else:
        num_features.append(unique_rows.index[i])
print('dataset has {} numerical and {} categorical features'.format(len(num_features), len(cat_features)))
df.nunique().sort_values()
df.describe()
(r, c) = orig_df.shape
df.drop_duplicates(inplace=True, ignore_index=True)
if df.shape == (r, c):
    print('ther is no duplicates rows')
else:
    print('no. of duplicates droped is ', r - df.shape[0])
df1 = df.copy()
for i in [i for i in df1.columns]:
    if df1[i].nunique() >= 12:
        Q1 = df1[i].quantile(0.25)
        Q3 = df1[i].quantile(0.75)
        IQR = Q3 - Q1
        df1 = df1[df1[i] <= Q3 + 1.5 * IQR]
        df1 = df1[df1[i] >= Q1 - 1.5 * IQR]
df1 = df1.reset_index(drop=True)
print('\n\x1b[1mInference:\x1b[0m Before removal of outliers, The dataset had {} samples.'.format(df.shape[0]))
print('\x1b[1mInference:\x1b[0m After removal of outliers, The dataset now has {} samples.'.format(df1.shape[0]))
df2 = df1.copy()
df2[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df2[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
null_df = pd.DataFrame(df2.isnull().sum().sort_values(), columns=['null_sum'])
null_df['null_percentage'] = round(null_df['null_sum'] / df2.shape[0], 2) * 100
print(null_df)
pass
pass
pass
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X = df2[['SkinThickness', 'Insulin']]
y = df2[target]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=2)
X_train.tail()
knn = KNNImputer(n_neighbors=5, weights='distance')
X_train_trf = knn.fit_transform(X_train)
X_test_trf = knn.transform(X_test)
lr = LogisticRegression()