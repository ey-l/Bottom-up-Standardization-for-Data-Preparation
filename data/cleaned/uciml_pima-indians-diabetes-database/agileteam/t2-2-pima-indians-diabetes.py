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
import pandas as pd
(X_train.shape, y_train.shape, X_test.shape)
X_train.head()
y_train.value_counts()
X_train.info()
X_train.isnull().sum()
X_test.isnull().sum()
X_train.describe()
print('Glucose:', len(X_train[X_train['Glucose'] == 0]))
print('BloodPressure:', len(X_train[X_train['BloodPressure'] == 0]))
print('SkinThickness:', len(X_train[X_train['SkinThickness'] == 0]))
print('Insulin:', len(X_train[X_train['Insulin'] == 0]))
print('BMI:', len(X_train[X_train['BMI'] == 0]))
print('Glucose:', len(X_test[X_test['Glucose'] == 0]))
print('BloodPressure:', len(X_test[X_test['BloodPressure'] == 0]))
print('SkinThickness:', len(X_test[X_test['SkinThickness'] == 0]))
print('Insulin:', len(X_test[X_test['Insulin'] == 0]))
print('BMI:', len(X_test[X_test['BMI'] == 0]))
del_idx = X_train[X_train['Glucose'] == 0].index
del_idx
print('Glucose 이상치 삭제 전 :', X_train.shape, y_train.shape)
X_train = X_train.drop(index=del_idx, axis=0)
y_train = y_train.drop(index=del_idx, axis=0)
print('Glucose 이상치 삭제 후 :', X_train.shape, y_train.shape)
cols = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
cols_mean = X_train[cols].mean()
X_train[cols].replace(0, cols_mean)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X_train[cols] = scaler.fit_transform(X_train[cols])
X_test[cols] = scaler.fit_transform(X_test[cols])
X = X_train.drop('id', axis=1)
test = X_test.drop('id', axis=1)
from sklearn.svm import SVC
model = SVC(random_state=2022)