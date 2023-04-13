import pandas as pd
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
from sklearn.model_selection import train_test_split
(X_train, X_test) = train_test_split(df, test_size=0.2, shuffle=True, random_state=2021)
y_train = X_train['Outcome']
X_train = X_train.drop(columns=['Outcome'])
y_test = X_test['Outcome']
X_test = X_test.drop(columns=['Outcome'])
(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
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
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
pd.DataFrame(X_train_scaled, columns=X_train.columns).head()
from sklearn.svm import SVC
model = SVC(random_state=2022)