import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.info()
df.describe()
zero_value = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for i in zero_value:
    df[i].replace(0, np.nan, inplace=True)
df.describe()
df = df.drop(['Insulin'], axis=1)
df.info()
count1 = pd.concat([df.isnull().sum()], axis=1, keys=['df'])
count1[count1.sum(axis=1) > 0]

def cap_data(df):
    for col in df.columns:
        print('capping the ', col)
        if (df[col].dtype == 'float64') | (df[col].dtype == 'int64'):
            percentiles = df[col].quantile([0.01, 0.99]).values
            df[col][df[col] <= percentiles[0]] = percentiles[0]
            df[col][df[col] >= percentiles[1]] = percentiles[1]
        else:
            df[col] = df[col]
    return df
final_df = cap_data(df)
sns.boxplot(data=df, x=df['BloodPressure'])
sns.boxplot(data=df, x=df['SkinThickness'])
sns.boxplot(data=df, x=df['Pregnancies'])
sns.boxplot(data=df, x=df['Glucose'])
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X = imp.fit_transform(df)
df2 = pd.DataFrame(X, columns=df.columns)
df2['Outcome'] = df2['Outcome'].round().astype('int64')
df2.describe()
sns.boxplot(data=df2, x=df2['BloodPressure'])
sns.boxplot(data=df, x=df['SkinThickness'])
sns.boxplot(data=df, x=df['Glucose'])
sns.boxplot(data=df, x=df['Pregnancies'])
df2.head(10)
df2.info()
count2 = pd.concat([df2.isnull().sum()], axis=1, keys=['df2'])
count2[count2.sum(axis=1) > 0]
sns.histplot(data=df2, x=df['Glucose'])
sns.histplot(data=df2, x=df2['BMI'])
sns.histplot(data=df2, x=df2['Age'])
pr_lab = ['veryLow', 'Low', 'Medium', 'High', 'superHigh']
df2['AgeLavel'] = pr_bins = pd.cut(df2['Age'], bins=5, labels=pr_lab, precision=0)
sns.histplot(data=df2, x=df2['AgeLavel'])
(fig, ax) = plt.subplots(figsize=(12, 6))
sns.countplot(hue='Outcome', x='AgeLavel', data=df2, ax=ax)
df2.tail(10)
df2 = df2.drop(['AgeLavel'], axis=1)
df2.head()
corr_matrix = df2.corr()
corr_matrix['Outcome'].sort_values(ascending=False)
plt.figure(figsize=(12, 20))
sns.heatmap(df2.corr(), annot=True, linewidths=2)

plt.figure(figsize=(20, 20))
sns.pairplot(data=df2, hue='Outcome', diag_kind='hist')

df2['Outcome'].value_counts()
df2.shape
df2.head()
corr_matrix = df2.corr()
corr_matrix['Outcome'].sort_values(ascending=False)
plt.figure(figsize=(12, 20))
sns.heatmap(df2.corr(), annot=True, linewidths=2)

df2.head()
X = df2.iloc[:, :-1].to_numpy()
y = df2.iloc[:, -1].to_numpy()
print(X.shape)
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state=0)
(X_r, y_r) = smk.fit_resample(X, y)
x = X_r
y = y_r
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train.shape)
print(x_train)
print(y_train.shape)
print(y_train)
print(x_test.shape)
print(x_test)
print(y_test.shape)
print(y_test)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
import sklearn.pipeline
from sklearn.pipeline import Pipeline
my_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('std_scaler', StandardScaler())])