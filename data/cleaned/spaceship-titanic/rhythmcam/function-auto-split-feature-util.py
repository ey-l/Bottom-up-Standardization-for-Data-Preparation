import numpy as np
import pandas as pd
TRAIN_PATH = 'data/input/spaceship-titanic/train.csv'
TEST_PATH = 'data/input/spaceship-titanic/test.csv'

def addSplitedColumn(df, COL, SPLIT_SIZE, SEP):
    for i in range(SPLIT_SIZE):
        df[COL + '_' + str(i)] = df[COL].str.split(pat=SEP, expand=True)[i]
    df = df.drop([COL], axis=1)
    return df
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
COL = 'Cabin'
SPLIT_SIZE = 3
SEP = '/'
train = addSplitedColumn(train, COL, SPLIT_SIZE, SEP)
train.head()
test = addSplitedColumn(test, COL, SPLIT_SIZE, SEP)
test.head()