import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataset.head(6)
dataset.info()
dataset.hist(figsize=(12, 12))
import scipy.stats as stats
import pylab
columns = list(dataset.columns.values)
columns

def make_qqplot(data, features):
    for value in features:
        print(f'\x1b[1m{value}\x1b[0m\n')
        figure = stats.probplot(dataset[value], dist='norm', plot=pylab)
make_qqplot(dataset, columns)
from scipy.stats import shapiro

def normtest_shapiro(data, features):
    for (index, value) in enumerate(columns):
        pval = shapiro(dataset[value]).pvalue
        alpha = 0.05
        if pval > alpha:
            print(index, f'\x1b[1m{value}\x1b[0m is normally distributed')
        elif pval < alpha:
            print(index, f'\x1b[1m{value}\x1b[0m is Not normally distributed')
normtest_shapiro(dataset, columns)
from scipy.stats import normaltest

def normtest_dAgostino(data, features):
    for (index, value) in enumerate(columns):
        pval = normaltest(dataset[value]).pvalue
        alpha = 0.05
        if pval > alpha:
            print(index, f'\x1b[1m{value}\x1b[0m is normally distributed')
        elif pval < alpha:
            print(index, f'\x1b[1m{value}\x1b[0m is Not normally distributed')
normtest_dAgostino(dataset, columns)