import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
t = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
t.head(10)
plt.hist(t.Glucose, bins=80)
plt.hist(t.Insulin, bins=80)
plt.hist(t.SkinThickness, bins=80)
plt.hist(t.BMI, bins=80)

def replace_0(df, col):
    df1 = df.copy()
    n = df.shape[0]
    m = df[col].mean()
    s = df[col].std()
    for i in range(768):
        if df.loc[i, col] == 0:
            df1.loc[i, col] = np.random.normal(m, s)
    return df1
t.count()[0]
t = replace_0(t, 'Glucose')
t.Insulin = t.Insulin.replace(to_replace=0, value=t.Insulin.mean())
t.SkinThickness = t.SkinThickness.replace(to_replace=0, value=t.SkinThickness.mean())
t.BloodPressure = t.BloodPressure.replace(to_replace=0, value=t.BloodPressure.mean())
t.BMI = t.BMI.replace(to_replace=0, value=t.BMI.mean())
plt.hist(t.Glucose, bins=80)
plt.hist(t.Insulin, bins=80)
plt.hist(t.SkinThickness, bins=80)
plt.hist(t.BMI, bins=80)
sns.distplot(t.BMI, color='blue')
X = t.drop(['Outcome'], axis=1)
y = t.Outcome
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=10000)