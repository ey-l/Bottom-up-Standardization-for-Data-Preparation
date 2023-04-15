import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
X = df[df.columns[:8]]
y = df['Outcome']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()