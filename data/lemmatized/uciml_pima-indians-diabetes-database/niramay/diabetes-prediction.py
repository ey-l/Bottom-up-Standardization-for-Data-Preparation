import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
all_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
all_data