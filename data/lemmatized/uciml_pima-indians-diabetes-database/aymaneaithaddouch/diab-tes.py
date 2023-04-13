import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import *
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.sample(3)
df.Outcome.value_counts()
df.Pregnancies.value_counts()
from sklearn.impute import KNNImputer
imputer = KNNImputer(missing_values=0)