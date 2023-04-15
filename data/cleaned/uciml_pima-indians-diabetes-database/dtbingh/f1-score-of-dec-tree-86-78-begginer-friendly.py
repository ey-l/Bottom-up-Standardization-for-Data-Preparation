import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.isna().sum()
df.hist(figsize=(20, 20))
dfBMI = df['BMI']
dfBMI.sort_values()
dfBMI[dfBMI == 0]
dfBP = df['BloodPressure']
df[df['BloodPressure'] == 0]
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
from pandas_profiling import ProfileReport
profile = ProfileReport(df)
profile
df['BloodPressure'].isna().sum()
import statistics
InsulinNull = df.loc[df['Insulin'].isnull()]
InsulinH = df.loc[df['Insulin'].notnull() & (df['Outcome'] == 0)]
InsulinH['Insulin'].median()
InsulinNullH = df.loc[df['Insulin'].isnull() & (df['Outcome'] == 0)]
InsulinNullH['Insulin'].fillna(102.5, inplace=True)
df.loc[(df['Outcome'] == 0) & df['Insulin'].isnull(), 'Insulin'] = 102.5
df.loc[(df['Outcome'] == 1) & df['Insulin'].isnull(), 'Insulin'] = 169.5
dfcopy = df

def Filler(col):
    Healthy = df
    Unhealthy = df
    Healthy = df.loc[df[col].notnull() & (df['Outcome'] == 0)]
    HealthyMedian = Healthy[col].median()
    Unhealthy = df.loc[df[col].notnull() & (df['Outcome'] == 1)]
    UnhealthyMedian = Unhealthy[col].median()
    df.loc[(df['Outcome'] == 0) & df[col].isnull(), col] = HealthyMedian
    df.loc[(df['Outcome'] == 1) & df[col].isnull(), col] = UnhealthyMedian
Filler('BMI')
Filler('SkinThickness')
Filler('BloodPressure')
Filler('Glucose')
profile2 = ProfileReport(df)
profile2
plt.style.use('ggplot')
(f, ax) = plt.subplots(figsize=(11, 15))
ax.set_facecolor('#fafafa')
ax.set(xlim=(-0.5, 300))
plt.ylabel('Variables')
plt.title('Overview Data Set')
ax = sns.boxplot(data=df, orient='h', palette='Set2')
lr = LinearRegression()
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=42, test_size=0.15)