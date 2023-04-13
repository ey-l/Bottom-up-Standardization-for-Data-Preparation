import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
tf.__version__
train_csv_path: str = 'data/input/titanic/train.csv'
data: pd.DataFrame = pd.read_csv(train_csv_path)
data.info()
data.head()
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch'], axis=1)
data.head()
data.isnull().sum()
data = data.dropna(subset=['Embarked'], inplace=False)
data['Age'] = data['Age'].fillna(data['Age'].mean(), inplace=False)
data.isnull().sum()
sex_col = data['Sex'] == 'male'
sex_col = sex_col.astype('int32')
data = data.drop(['Sex'], axis=1)
data['Sex'] = sex_col
data.head()
data = pd.get_dummies(data, columns=['Embarked'])
data.head()
X = data.drop('Survived', axis=1).to_numpy()
y = data['Survived'].to_numpy()
(X.shape, y.shape)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
tf.random.set_seed(42)
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.2, random_state=42)
(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
tf.random.set_seed(42)
model_1 = tf.keras.Sequential([tf.keras.layers.Dense(9, activation='relu'), tf.keras.layers.Dense(15, activation='relu'), tf.keras.layers.Dense(50, activation='relu'), tf.keras.layers.Dense(2, activation='softmax')])
model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])