import numpy as np
import pandas as pd
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
cols = ['PassengerId', 'Sex', 'Age']
submission_all_dead = _input0[cols]
submission_all_dead['Survived'] = 0
submission_all_dead.head()
submission_all_live = _input0[cols]
submission_all_live['Survived'] = 1
submission_all_live.head()
submission_women_child = _input0[cols]
submission_women_child['Survived'] = 0
submission_women_child[submission_women_child['Sex'] == 'female']['Survived'] = 1
submission_women_child[submission_women_child['Age'] < 18]['Survived'] = 1