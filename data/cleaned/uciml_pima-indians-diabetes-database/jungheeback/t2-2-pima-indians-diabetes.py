import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def exam_data_load(df, target, id_name='', null_name=''):
    if id_name == '':
        df = df.reset_index().rename(columns={'index': 'id'})
        id_name = 'id'
    else:
        id_name = id_name
    if null_name != '':
        df[df == null_name] = np.nan
    (X_train, X_test) = train_test_split(df, test_size=0.2, random_state=2021)
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[target])
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[target])
    return (X_train, X_test, y_train, y_test)
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
(X_train, X_test, y_train, y_test) = exam_data_load(df, target='Outcome')
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train.info()

y_train.info()
col = list(X_train.columns)
for c in col:
    print(c, len(X_train.loc[X_train[c] == 0]))
col = list(X_test.columns)
for c in col:
    print(c, len(X_test.loc[X_test[c] == 0]))
X_train.loc[X_train['Pregnancies'] == 0, 'Pregnancies'] = X_train['Pregnancies'].mean()
X_train.loc[X_train['BloodPressure'] == 0, 'BloodPressure'] = X_train['BloodPressure'].mean()
X_train.loc[X_train['SkinThickness'] == 0, 'SkinThickness'] = X_train['SkinThickness'].mean()
X_train.loc[X_train['Insulin'] == 0, 'Insulin'] = X_train['Insulin'].mean()
X_train.loc[X_train['BMI'] == 0, 'BMI'] = X_train['BMI'].mean()
X_train.loc[X_train['Glucose'] == 0, 'Glucose'] = X_train['Glucose'].mean()
X_test.loc[X_test['Pregnancies'] == 0, 'Pregnancies'] = X_test['Pregnancies'].mean()
X_test.loc[X_test['BloodPressure'] == 0, 'BloodPressure'] = X_test['BloodPressure'].mean()
X_test.loc[X_test['SkinThickness'] == 0, 'SkinThickness'] = X_test['SkinThickness'].mean()
X_test.loc[X_test['Insulin'] == 0, 'Insulin'] = X_test['Insulin'].mean()
X_test.loc[X_test['BMI'] == 0, 'BMI'] = X_test['BMI'].mean()
col = list(X_train.columns)
for c in col:
    print(c, len(X_train.loc[X_train[c] == 0]))
col = list(X_test.columns)
for c in col:
    print(c, len(X_test.loc[X_test[c] == 0]))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature = ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
X_train[feature] = scaler.fit_transform(X_train[feature])
X_test[feature] = scaler.fit_transform(X_test[feature])
X_train['Pregnancies']
from sklearn.svm import SVC
model = SVC()