import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.columns
_input1 = _input1.drop(['PassengerId', 'Name'], axis=1)
np.sum(_input1.isna())
_input1 = _input1.dropna()
np.sum(_input1.isna())
_input1.HomePlanet.value_counts()
_input1.CryoSleep.value_counts()
_input1.VIP.value_counts()
_input1.Transported.value_counts()
HomePlanet_transported = pd.crosstab(_input1.HomePlanet, _input1.Transported)
HomePlanet_transported.plot(figsize=(20, 10), kind='bar')
plt.title('Relationship between Home Planet and Transported')
plt.xlabel('Home Planet')
plt.ylabel('Transported')
CryoSleep_transported = pd.crosstab(_input1.CryoSleep, _input1.Transported)
CryoSleep_transported.plot(figsize=(20, 10), kind='bar')
plt.title('Relationship between CryoSleep and Transported')
plt.xlabel('CryoSleep')
plt.ylabel('Transported')
VIP_transported = pd.crosstab(_input1.VIP, _input1.Transported)
VIP_transported.plot(figsize=(20, 10), kind='bar')
plt.title('Relationship between VIP and Transported')
plt.xlabel('VIP')
plt.ylabel('Transported')
age_transported = pd.crosstab(_input1.Age, _input1.Transported)
age_transported.plot(figsize=(20, 10), kind='bar')
plt.title('Relationship between Age and Transported')
plt.xlabel('Age')
plt.ylabel('Transported')
for col_name in _input1.columns:
    if _input1[col_name].dtype == 'object':
        _input1[col_name] = _input1[col_name].astype('category')
        _input1[col_name] = _input1[col_name].cat.codes
X = _input1.drop(['Transported'], axis=1)
y = _input1.Transported
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)