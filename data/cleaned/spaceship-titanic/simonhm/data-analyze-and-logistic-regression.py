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
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_data.columns
train_data = train_data.drop(['PassengerId', 'Name'], axis=1)
np.sum(train_data.isna())
train_data = train_data.dropna()
np.sum(train_data.isna())
train_data.HomePlanet.value_counts()
train_data.CryoSleep.value_counts()
train_data.VIP.value_counts()
train_data.Transported.value_counts()
HomePlanet_transported = pd.crosstab(train_data.HomePlanet, train_data.Transported)
HomePlanet_transported.plot(figsize=(20, 10), kind='bar')
plt.title('Relationship between Home Planet and Transported')
plt.xlabel('Home Planet')
plt.ylabel('Transported')
CryoSleep_transported = pd.crosstab(train_data.CryoSleep, train_data.Transported)
CryoSleep_transported.plot(figsize=(20, 10), kind='bar')
plt.title('Relationship between CryoSleep and Transported')
plt.xlabel('CryoSleep')
plt.ylabel('Transported')
VIP_transported = pd.crosstab(train_data.VIP, train_data.Transported)
VIP_transported.plot(figsize=(20, 10), kind='bar')
plt.title('Relationship between VIP and Transported')
plt.xlabel('VIP')
plt.ylabel('Transported')
age_transported = pd.crosstab(train_data.Age, train_data.Transported)
age_transported.plot(figsize=(20, 10), kind='bar')
plt.title('Relationship between Age and Transported')
plt.xlabel('Age')
plt.ylabel('Transported')
for col_name in train_data.columns:
    if train_data[col_name].dtype == 'object':
        train_data[col_name] = train_data[col_name].astype('category')
        train_data[col_name] = train_data[col_name].cat.codes

X = train_data.drop(['Transported'], axis=1)
y = train_data.Transported
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)