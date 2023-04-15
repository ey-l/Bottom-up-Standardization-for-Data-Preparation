import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train
train.info()
df_dc = pd.get_dummies(train, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP'])
df_dc.drop(['Name', 'Cabin', 'PassengerId'], axis=1, inplace=True)
df_dc
first_column = df_dc.pop('Transported')
df_dc.insert(16, 'Transported', first_column)
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df_dc['Transported'] = label_encoder.fit_transform(df_dc['Transported'])
df_dc
df_dc.info()
df_dc.fillna(df_dc[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].median(), inplace=True)
df_dc.info()
X = df_dc.iloc[:, :-1]
y = df_dc.iloc[:, -1]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.4, random_state=42)
logreg = LogisticRegression()