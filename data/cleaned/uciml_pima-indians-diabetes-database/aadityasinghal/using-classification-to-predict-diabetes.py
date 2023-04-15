import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataset.head()
dataset.isnull().sum()
sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')
dataset.info()
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values
imputer = SimpleImputer(missing_values=0, strategy='mean')