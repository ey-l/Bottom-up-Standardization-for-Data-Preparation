import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df
groups = []
pass_id = np.array(df['PassengerId'])
for i in range(len(pass_id)):
    temp = pass_id[i]
    pass_id[i] = int(temp[6:])
    groups.append(int(temp[:4]))
groups = pd.Series(np.array(groups))
df['PassengerId'] = pd.Series(pass_id.astype(int))
df.insert(loc=0, column='group', value=groups)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
temp = df.describe()
features = df.columns
numerical_features = temp.columns
for i in numerical_features:
    mean = df[i].mean()
    df[i] = df[i].fillna(value=mean)
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'Name', 'VIP', 'Transported', 'Cabin']
for i in categorical_features:
    df[i] = encoder.fit_transform(np.array(df[i]).reshape(-1, 1))
    median = df[i].median()
    df[i] = df[i].fillna(value=median)
df = df.drop(['Name'], axis='columns')
numerical_features
features = df.columns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler1 = StandardScaler()
scaler2 = MinMaxScaler()
df['Cabin'] = scaler1.fit_transform(np.array(df['Cabin']).reshape(-1, 1))
df['Cabin'] = scaler2.fit_transform(np.array(df['Cabin']).reshape(-1, 1))
for i in numerical_features:
    if i != 'PassengerId':
        df[i] = scaler1.fit_transform(np.array(df[i]).reshape(-1, 1))
        df[i] = scaler2.fit_transform(np.array(df[i]).reshape(-1, 1))
df
df.isna().sum()
features = df.columns
for i in features:
    sns.distplot(df[i])

sns.heatmap(df.corr(), annot=True)

def low_corr_features(df, target_variable, corr_val):
    import numpy as np
    features = df.columns
    corr_df = df.corr()
    return_list = []
    corr_with_target = np.array(corr_df[target_variable])
    for i in range(0, len(corr_with_target)):
        if np.absolute(corr_with_target[i]) < corr_val:
            return_list.append(features[i])
    return return_list
low_corr_features(df, 'Transported', 0.02)
from sklearn.model_selection import train_test_split
X = np.array(df.drop(['Transported'], axis='columns'))
y = np.array(df['Transported'])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
(model1, model2, model3, model4, model5) = (LogisticRegression(), RandomForestClassifier(), LinearDiscriminantAnalysis(), DecisionTreeClassifier(), GaussianNB())