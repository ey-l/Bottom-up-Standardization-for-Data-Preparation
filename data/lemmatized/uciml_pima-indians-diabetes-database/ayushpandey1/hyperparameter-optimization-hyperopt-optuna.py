import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
import numpy as np
df['Glucose'] = np.where(df['Glucose'] == 0, df['Glucose'].median(), df['Glucose'])
df.head(10)
X = df.drop('Outcome', axis=1)
y = df['Outcome']
pd.DataFrame(X, columns=df.columns[:-1])
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier