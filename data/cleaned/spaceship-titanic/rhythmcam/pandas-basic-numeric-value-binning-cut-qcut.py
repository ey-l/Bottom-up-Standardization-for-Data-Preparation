import numpy as np
import pandas as pd
TRAIN_PATH = 'data/input/spaceship-titanic/train.csv'
train = pd.read_csv(TRAIN_PATH)
train.columns

def checkNull_fillData(df):
    for col in df.columns:
        if len(df.loc[df[col].isnull() == True]) != 0:
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                df.loc[df[col].isnull() == True, col] = df[col].mean()
            else:
                df.loc[df[col].isnull() == True, col] = df[col].mode()[0]
checkNull_fillData(train)
train['Age']
DEVIDE_NUM = 10
train['Age_Cut'] = pd.cut(train['Age'].values, DEVIDE_NUM)
train['Age_Cut']
train['Age_Cut'].value_counts()
DEVIDE_NUM = 10
train['Age_Cut'] = pd.cut(train['Age'].values, DEVIDE_NUM, labels=np.arange(0, DEVIDE_NUM))
train['Age_Cut'] = train['Age_Cut'].astype(int)
train['Age_Cut']
train['Age_Cut'].value_counts()
DEVIDE_NUM = 10
train['Age_Cut'] = pd.cut(train['Age'].values, DEVIDE_NUM, labels=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
train['Age_Cut'] = train['Age_Cut'].astype(int)
train['Age_Cut']
train['Age_Cut'].value_counts()
DEVIDE_NUM = 10
train['Age_QCut'] = pd.qcut(train['Age'].values, DEVIDE_NUM)
train['Age_QCut']
train['Age_QCut'].value_counts()
DEVIDE_NUM = 10
train['Age_QCut'] = pd.qcut(train['Age'].values, DEVIDE_NUM, labels=np.arange(0, DEVIDE_NUM))
train['Age_QCut'] = train['Age_QCut'].astype(int)
train['Age_QCut']
train['Age_QCut'].value_counts()
DEVIDE_NUM = 10
train['Age_QCut'] = pd.qcut(train['Age'].values, DEVIDE_NUM, labels=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
train['Age_QCut'] = train['Age_QCut'].astype(int)
train['Age_QCut']
train['Age_QCut'].value_counts()

def age_group_function(age):
    if age <= 0:
        return 0
    elif (age > 0) & (age <= 10):
        return 1
    elif (age > 10) & (age <= 20):
        return 2
    elif (age > 20) & (age <= 30):
        return 3
    elif (age > 30) & (age <= 40):
        return 4
    elif (age > 40) & (age <= 50):
        return 5
    elif (age > 50) & (age <= 60):
        return 6
    elif (age > 60) & (age <= 70):
        return 7
    elif (age > 70) & (age <= 80):
        return 8
    elif (age > 80) & (age <= 90):
        return 9
    else:
        return 10
train['Age_Generation'] = train['Age'].apply(age_group_function)
train['Age_Generation']
train['Age_Generation'].value_counts()