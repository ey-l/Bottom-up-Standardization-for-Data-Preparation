import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.describe().transpose()
data.info()
pass
pass
data.corr()
data_null = pd.DataFrame(data.isnull().sum(), columns=['Number Of Null'])
data_null['Percentage Of Null'] = data_null['Number Of Null'] / len(data)
data_null
data[data.duplicated()]
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=True)
print('X_train Shape :', X_train.shape)
print('X_test Shape :', X_test.shape)
print('y_train Shape :', y_train.shape)
print('y_test Shape :', y_test.shape)
RandomForestClassifierModel = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=8)