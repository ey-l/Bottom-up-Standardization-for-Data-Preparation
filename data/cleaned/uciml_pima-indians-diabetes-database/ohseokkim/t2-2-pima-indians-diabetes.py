import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def exam_data_load(df, target, id_name='', null_name=''):
    if id_name == '':
        df = df.reset_index().rename(columns={'index': 'id'})
        id_name = 'id'
    else:
        id_name = id_name
    if null_name != '':
        df[df == null_name] = np.nan
    (X_train, X_test) = train_test_split(df, test_size=0.2, random_state=2021)
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[target])
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[target])
    return (X_train, X_test, y_train, y_test)
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
(X_train, X_test, y_train, y_test) = exam_data_load(df, target='Outcome')
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
import pandas as pd
import sklearn
help(sklearn)
import sklearn.ensemble
dir(sklearn.ensemble)
from sklearn.ensemble import RandomForestClassifier
X_train.head().T
X_train.info()
X_train.describe()
X_train.isnull().sum()
X_test.isnull().sum()
help(df.drop)
X_train.drop('id', axis=1, inplace=True)
X_test.drop('id', axis=1, inplace=True)
X_train.info()
help(RandomForestClassifier)
model = RandomForestClassifier()