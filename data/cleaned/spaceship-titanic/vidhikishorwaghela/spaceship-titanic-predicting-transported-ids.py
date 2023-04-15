import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
X = train_data[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
y = train_data['Transported']
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=42)
preprocess = make_column_transformer((SimpleImputer(strategy='most_frequent'), ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']), (OneHotEncoder(handle_unknown='ignore'), ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']))
pipeline = make_pipeline(preprocess, RandomForestClassifier(n_estimators=100, random_state=42))