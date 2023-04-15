import matplotlib.pyplot as matplotlib
import seaborn
import pandas
import numpy


def reshape(list1D):
    return numpy.array(list1D).reshape(-1, 1)

def plot_ours(model):
    x = numpy.linspace(0, 1, 50)
    y = model.predict(reshape(x))
    matplotlib.figure(figsize=(4, 4))
    matplotlib.plot(x, y, color='red')
    matplotlib.suptitle('Our Logistic model')
    matplotlib.xlabel('x')
    matplotlib.ylabel('y')
    matplotlib.show()

def plot_lr():
    logistical = lambda x: numpy.exp(x) / (1 + numpy.exp(x))
    x = numpy.linspace(-10, 10, 50)
    y = logistical(x)
    matplotlib.figure(figsize=(4, 4))
    matplotlib.plot(x, y, color='red')
    matplotlib.suptitle('Logisitc Regression model')
    matplotlib.xlabel('x')
    matplotlib.ylabel('y')
    matplotlib.show()
plot_lr()
from sklearn.linear_model import LogisticRegression as lr
x = [0.4, 0.1, 0.7, 0.04, 0.99, 3e-05, 0.863, 0.65, 0.72, 0.34, 0.51, 0.49]
y = [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0]
model = lr()