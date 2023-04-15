import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]
TRAIN_PATH = 'data/input/spaceship-titanic/train.csv'
ID = 'PassengerId'
TARGET = 'Transported'
train = pd.read_csv(TRAIN_PATH)
str_col_list = train.describe(include='O').columns.tolist()
str_col_list
from sklearn.preprocessing import LabelEncoder
for col in str_col_list:
    label_encoder = LabelEncoder()
    train[col] = label_encoder.fit_transform(train[col])
FI = np.abs(train.corr()[TARGET]).sort_values()
df_FI = pd.DataFrame(FI)
LastFI = df_FI[::-1].T.drop([ID, TARGET], axis=1)
LastFI.T
plt.bar(LastFI.T.index, LastFI.T[TARGET])
plt.title('Feature Importance', fontsize=20)
plt.xlabel('Feature', fontsize=18)
plt.ylabel('Importance', fontsize=18)
plt.xticks(LastFI.T.index, LastFI.T.index, fontsize=10)
