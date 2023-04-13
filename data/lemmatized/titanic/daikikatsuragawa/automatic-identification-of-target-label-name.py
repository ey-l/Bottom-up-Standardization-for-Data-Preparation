import pandas as pd
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
train_columns = _input1.columns.to_list()
test_columns = _input0.columns.to_list()
tmp = list(set(train_columns) - set(test_columns))
label = tmp[0]
print(label)
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
train = TabularDataset('data/input/titanic/train.csv')
test = TabularDataset('data/input/titanic/test.csv')
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
train_columns = _input1.columns.to_list()
test_columns = _input0.columns.to_list()
tmp = list(set(train_columns) - set(test_columns))
label = tmp[0]
time_limit = 60