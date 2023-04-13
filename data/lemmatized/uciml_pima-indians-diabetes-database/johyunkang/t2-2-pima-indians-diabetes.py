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
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print('Train Feature:\n', X_train[X_train[zero_features] == 0].count())
print('    ')
print('Ttest Feature:\n', X_test[X_test[zero_features] == 0].count())
df_mean = X_train[X_train[zero_features] != 0][zero_features].mean()
print(df_mean)
X_train[zero_features] = X_train[zero_features].replace(0, df_mean)
print(X_train[X_train[zero_features] == 0].count())
df_mean_test = X_test[X_test[zero_features] != 0][zero_features].mean()
print(df_mean_test)
X_test[zero_features] = X_test[zero_features].replace(0, df_mean_test)
print(X_test[X_test[zero_features] == 0].count())
X_test.describe()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.model_selection import train_test_split
(x_tr, x_te, y_tr, y_te) = train_test_split(X_train_scaled, y_train, test_size=0.3, random_state=123)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=123)