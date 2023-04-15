import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import ensemble
data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
enc = LabelEncoder()