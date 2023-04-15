import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.isnull().sum()
df.drop(['Outcome'], axis=1).hist(bins=50, figsize=(20, 15))

miss_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
df[miss_columns].isin([0]).sum()
df[miss_columns] = df[miss_columns].replace(0, np.NaN)
df.isnull().sum()
target_df = df['Outcome']
print('Percentage of 0 (non-diabetic): ', np.round(1 - sum(target_df) / len(target_df), 3))
print('Percentage of 1 (diabetic): ', np.round(sum(target_df) / len(target_df), 3))
target_df.hist()

corrmatrix = df.corr()
sns.heatmap(corrmatrix, annot=True)

p = sns.pairplot(df, hue='Outcome')

from sklearn.model_selection import train_test_split
target_df = df['Outcome']
features_df = df.drop(['Outcome'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(features_df, target_df, test_size=0.3, random_state=42, stratify=target_df)
print('% of 0 (non-diabetic) on train set: ', np.round(1 - sum(y_train) / len(y_train), 3))
print('% of 0 (non-diabetic) on test set: ', np.round(1 - sum(y_test) / len(y_test), 3))
print('% of 1 (diabetic) on train set: ', np.round(sum(y_train) / len(y_train), 3))
print('% of 1 (diabetic) on test set: ', np.round(sum(y_test) / len(y_test), 3))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score

def make_pipeline(classifier):
    """Create a pipeline for a classifier."""
    steps = [('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler()), ('classifier', classifier)]
    return Pipeline(steps)

def score_model(pipeline, test_set):
    """Score a given pipeline using Accuracy, ROC AUC, Precision and Recall"""
    predict = pipeline.predict(test_set)
    predict_proba = pipeline.predict_proba(test_set)[:, 1]
    metrics = {'accuracy': pipeline.score(test_set, y_test), 'roc_auc': roc_auc_score(y_test, predict_proba), 'precision': precision_score(y_test, predict), 'recall': recall_score(y_test, predict)}
    return metrics

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
pipe_rfc = make_pipeline(rfc)