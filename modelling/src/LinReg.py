'''
Created on 18 Oct. 2017

@author: user
'''
from sklearn import linear_model
import numpy as np
import pandas as pd

pcaXtrain = pd.read_csv('pcaReducedTrain.csv').values
pcaXtest = pd.read_csv('pcaReducedTest.csv').values
borutaXtrain = pd.read_csv('borutaTrain.csv').values
borutaXtest = pd.read_csv('borutaTest.csv').values
fcbfXtrain = pd.read_csv('fcbfTrain.csv').values
fcbfXtest = pd.read_csv('fcbfTest.csv').values
y_train = pd.read_csv('YtrainKNNCat.csv').values
y_test = pd.read_csv('YtestKNNCat.csv').values

regr = linear_model.LinearRegression()

regr.fit(fcbfXtrain, y_train)

print(regr.score(fcbfXtest, y_test))