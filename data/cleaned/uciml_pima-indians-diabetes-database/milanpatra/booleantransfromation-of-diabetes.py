import pandas as pd
import numpy as np
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.sample(6)
url = 'https://raw.githubusercontent.com/adityakumar529/Coursera_Capstone/master/diabetes.csv'
data = pd.read_csv(url)
data.sample(6)
data.info()
data.describe()
data.columns
colsToModify = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in colsToModify:
    print(col + ' - ')
    print(data[data[col] == 0][col].value_counts())
print('Count of zero entries in this column')
colsToModify = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
val = []
for col in colsToModify:
    val.append(len(data[data[col] == 0]))
zeroCount = pd.DataFrame(val, index=colsToModify, columns=['zeroCount'])
zeroCount
for col in colsToModify:
    data[col] = data[col].replace(0, np.NaN)
    mean = int(data[col].mean(skipna=True))
    data[col] = data[col].replace(np.NaN, mean)
data.describe()
data.sample(10)
df = data.drop([data.columns[-1]], axis=1)
meanList = df.mean()
mean_threshold = meanList.to_dict()
mean_threshold
df1 = df.copy()
df1['BMI'] = df1['BMI'] >= df1['BMI'].mean()
df1.head(10)

def df_toBoolean(df, col, th):
    df[col] = df[col] >= th
    return df
df1 = df.copy()
for col in df1.columns:
    df1 = df_toBoolean(df1, col, df1[col].mean())
df1.head()

def changedf_to_ZerosOnes(df, col):
    df[col] = df[col].astype(int)
    return df
finaldf1 = df1.copy()
for col in mean_threshold.keys():
    changedf_to_ZerosOnes(finaldf1, col)
finaldf1.head()
finaldata = finaldf1.copy()
finaldata['Outcome'] = data['Outcome']
finaldata.head()

def countOnesZeros(df):
    dmy = pd.DataFrame()
    for col in df.columns:
        cnt = df[col].value_counts()
        dmy[col] = pd.DataFrame(cnt)
    return dmy
ans = countOnesZeros(finaldata)
answer = ans.set_axis(['Ones', 'Twos'], axis=0)
answer
answer.T