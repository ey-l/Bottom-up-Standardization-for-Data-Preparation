import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(f'shape of the Diabetes dataset :- {dataset.shape}')
print('\n ***************************** \n')
print(f'Sample Dataset :- \n {dataset.head()}')
print('\n ***************************** \n')
print(f'checking for null values :- \n {dataset.isnull().sum()}')
print('\n ***************************** \n')
print(f'Number of Duplicate values :- {len(dataset.loc[dataset.duplicated()])}')
pass
pass
pass
X = dataset.drop(['Outcome'], axis=1)
y = dataset['Outcome']
col_names = list(X.columns)
pipeline = Pipeline([('std_scale', PowerTransformer(method='yeo-johnson'))])
X = pd.DataFrame(pipeline.fit_transform(X), columns=col_names)
print(X.head())
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1, random_state=42)
print(f'Size Of The Train Dataset :- {len(X_train)}')
print(f'Size Of The Test Dataset :- {len(X_test)}')
train_scores = []
test_scores = []
for i in range(1, 25):
    knn_clf = KNeighborsClassifier(n_neighbors=i)