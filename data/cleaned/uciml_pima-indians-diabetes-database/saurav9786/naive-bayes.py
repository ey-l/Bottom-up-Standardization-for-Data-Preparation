import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pima_indians_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
pima_indians_df.describe().T
pima_indians_df['Outcome'].value_counts()
pima_indians_df['SkinThickness'].value_counts(0)
pima_indians_df['Insulin'].value_counts()
import random
pima_indians_df['SkinThickness'] = pima_indians_df['SkinThickness'].replace(0, random.randrange(30, 40))
pima_indians_df['Insulin'] = pima_indians_df['Insulin'].replace(0, random.randrange(30, 140))
pima_indians_df.head(10)
array = pima_indians_df.values
X = array[:, 0:8]
Y = array[:, 8]
test_size = 0.15
seed = 7
(X_train, X_test, Y_train, Y_test) = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model = GaussianNB()