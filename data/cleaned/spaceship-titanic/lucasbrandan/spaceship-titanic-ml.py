import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
explo_path = 'data/input/spaceship-titanic/train.csv'
explo_data = pd.read_csv(explo_path)
explo_data.head()
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
spaceship_file_path = 'data/input/spaceship-titanic/train.csv'
spaceship_data = pd.read_csv(spaceship_file_path)
y = spaceship_data.Transported
features = ['Age', 'CryoSleep', 'VIP']
spaceship_data['CryoSleep'] = spaceship_data['CryoSleep'].replace(['True', 'False'], [1, 0])
spaceship_data['VIP'] = spaceship_data['VIP'].replace(['True', 'False'], [1, 0])
X = spaceship_data[features]
my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
(train_X, val_X, train_y, val_y) = train_test_split(X, y, random_state=1)
rf_model = RandomForestRegressor(random_state=1)