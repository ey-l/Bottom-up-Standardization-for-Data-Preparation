import numpy as np
import pandas as pd
import seaborn as sb
sb.set_style('dark')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


def evaluate_classifier(clf, data, target, split_ratio):
    (trainX, testX, trainY, testY) = train_test_split(data, target, train_size=split_ratio, random_state=0)