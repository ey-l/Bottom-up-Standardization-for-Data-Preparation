import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

def plot_feature_importance(estimator_object, X_train, y_train):
    model = estimator_object