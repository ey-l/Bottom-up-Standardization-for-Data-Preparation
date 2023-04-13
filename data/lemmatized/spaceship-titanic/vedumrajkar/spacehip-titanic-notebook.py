import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
groups = []
pass_id = np.array(_input1['PassengerId'])
for i in range(len(pass_id)):
    temp = pass_id[i]
    pass_id[i] = int(temp[6:])
    groups.append(int(temp[:4]))
groups = pd.Series(np.array(groups))
_input1['PassengerId'] = pd.Series(pass_id.astype(int))
_input1.insert(loc=0, column='group', value=groups)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
temp = _input1.describe()
features = _input1.columns
numerical_features = temp.columns
for i in numerical_features:
    mean = _input1[i].mean()
    _input1[i] = _input1[i].fillna(value=mean)
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'Name', 'VIP', 'Transported', 'Cabin']
for i in categorical_features:
    _input1[i] = encoder.fit_transform(np.array(_input1[i]).reshape(-1, 1))
    median = _input1[i].median()
    _input1[i] = _input1[i].fillna(value=median)
_input1 = _input1.drop(['Name'], axis='columns')
numerical_features
features = _input1.columns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler1 = StandardScaler()
scaler2 = MinMaxScaler()
_input1['Cabin'] = scaler1.fit_transform(np.array(_input1['Cabin']).reshape(-1, 1))
_input1['Cabin'] = scaler2.fit_transform(np.array(_input1['Cabin']).reshape(-1, 1))
for i in numerical_features:
    if i != 'PassengerId':
        _input1[i] = scaler1.fit_transform(np.array(_input1[i]).reshape(-1, 1))
        _input1[i] = scaler2.fit_transform(np.array(_input1[i]).reshape(-1, 1))
_input1
_input1.isna().sum()
features = _input1.columns
for i in features:
    sns.distplot(_input1[i])
sns.heatmap(_input1.corr(), annot=True)

def low_corr_features(df, target_variable, corr_val):
    import numpy as np
    features = _input1.columns
    corr_df = _input1.corr()
    return_list = []
    corr_with_target = np.array(corr_df[target_variable])
    for i in range(0, len(corr_with_target)):
        if np.absolute(corr_with_target[i]) < corr_val:
            return_list.append(features[i])
    return return_list
low_corr_features(_input1, 'Transported', 0.02)
from sklearn.model_selection import train_test_split
X = np.array(_input1.drop(['Transported'], axis='columns'))
y = np.array(_input1['Transported'])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
(model1, model2, model3, model4, model5) = (LogisticRegression(), RandomForestClassifier(), LinearDiscriminantAnalysis(), DecisionTreeClassifier(), GaussianNB())