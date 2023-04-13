import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df.shape
pass
pass
df.isnull().sum()
pass
pass
pass
pass
pass
pass
from sklearn.model_selection import train_test_split
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.1, random_state=0)
x_train.shape
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()