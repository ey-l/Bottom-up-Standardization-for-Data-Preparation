import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier as xgbc
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_targets = _input1.pop('Transported').astype('int64')
_input1.head(3)
params = dict(base_score=None, booster=None, colsample_bylevel=0.7, colsample_bynode=0.7, colsample_bytree=0.7, enable_categorical=False, gamma=0, gpu_id=None, importance_type=None, interaction_constraints=None, learning_rate=0.001, max_delta_step=None, max_depth=10, min_child_weight=None, missing=np.nan, monotone_constraints=None, n_estimators=10000, n_jobs=-1, num_parallel_tree=None, predictor=None, random_state=42, reg_alpha=1, reg_lambda=1, scale_pos_weight=None, subsample=None, tree_method=None, validate_parameters=None, verbosity=0)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = []
models = []
for (fold, (train_idx, val_idx)) in enumerate(skf.split(_input1, train_targets)):
    X_train = _input1.iloc[train_idx]
    X_val = _input1.iloc[val_idx]
    y_train = train_targets[train_idx]
    y_val = train_targets[val_idx]
    model = xgbc(**params)