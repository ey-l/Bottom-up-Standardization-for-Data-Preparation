import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.rename(columns={'DiabetesPedigreeFunction': 'DiabPed', 'BloodPressure': 'BP', 'SkinThickness': 'SkinThick', 'Pregnancies': 'Preg'}, inplace=True)
df.head(10)
X = df.drop('Outcome', axis=1)
Y = df['Outcome']
Y.value_counts()
sns.heatmap(df.isnull(), yticklabels=False, cmap='YlGnBu')
X.describe()
X = X.replace(0, np.NaN)
X.describe()
X.hist(stacked=False, bins=40, figsize=(12, 40), layout=(14, 2), color='blue')
from sklearn.impute import SimpleImputer
replace_0 = SimpleImputer(missing_values=np.NaN, strategy='mean')
cols = X.columns
X = pd.DataFrame(replace_0.fit_transform(X))
X.columns = cols
l = X.head(10)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.1, random_state=0)
x_train.shape
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')