import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
pass
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')
diabetes_data = pd.read_csv('diabetes.csv')
diabetes_data.head()
diabetes_data.shape
diabetes_data.describe()
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=1 / 5, shuffle=False)
classifier = svm.SVC(kernel='linear')