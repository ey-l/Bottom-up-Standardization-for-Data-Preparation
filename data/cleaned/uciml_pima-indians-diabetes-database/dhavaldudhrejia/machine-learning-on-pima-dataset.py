import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe()
df.isnull().sum()
cormat = df.corr()
plt.figure(figsize=(20, 20))
g = sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')
df.Outcome.value_counts()
from sklearn.model_selection import train_test_split
X = df.drop('Outcome', axis=1)
y = df.Outcome
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=10)
df.value_counts(df['Pregnancies'] == 0)
from sklearn.impute import SimpleImputer
fill_values = SimpleImputer(missing_values=0, strategy='mean')
X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)