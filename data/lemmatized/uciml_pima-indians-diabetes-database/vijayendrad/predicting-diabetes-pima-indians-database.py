import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head(10)
df.info()
features = list(df.columns)
features
import seaborn as sn
import matplotlib.pyplot as plt
pass
ax = sn.countplot(x='Outcome', data=df)
sn.pairplot(data=df, hue='Outcome')
df = df.dropna()
df.Outcome.value_counts()
from sklearn.utils import resample
no_diabetes = df[df.Outcome == 0]
yes_diabetes = df[df.Outcome == 1]
df_minority_upsampled = resample(yes_diabetes, replace=True, n_samples=250)
new_df = pd.concat([no_diabetes, df_minority_upsampled])
new_df.info()
from sklearn.utils import shuffle
new_df = shuffle(new_df)
X = new_df[features]
y = new_df['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=123)
from sklearn.tree import DecisionTreeClassifier
clf_tree = DecisionTreeClassifier(max_depth=5, criterion='gini')