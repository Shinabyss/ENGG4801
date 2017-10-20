from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import *
from boto.s3.multidelete import Deleted
import statsmodels
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
from statsmodels.imputation import mice
import statsmodels.api as sm
from sklearn.datasets import load_iris
import numpy as numpy
from knnimpute import (
    knn_impute_few_observed,
    knn_impute_with_argpartition,
    knn_impute_optimistic,
    knn_impute_reference,
)

dataRaw = pd.read_csv('ReducedDataNoLabel.csv')
#data = dataRaw.set_index(['label'])

train, test = train_test_split(dataRaw, test_size=0.2)

train_imputed = knn_impute_reference(train.as_matrix(), np.isnan(train.as_matrix()), k=5)

test_imputed = knn_impute_reference(test.as_matrix(), np.isnan(test.as_matrix()), k=5)
#imp = mice.MICEData(data)

#fml = 'y ~ '

#for i in list(imp.data):
#    fml = fml + i + ' + '
    
#fml = fml[:-3]

#mice = mice.MICE(fml, sm.OLS, imp)
#results = mice.fit(10, 10)
#print(results.summary())

#j=0

print(train_imputed)

print(test_imputed)

numpy.savetxt("trainKNN.csv", train_imputed, delimiter=",")

numpy.savetxt("testKNN.csv", test_imputed, delimiter=",")

#for j in range(20):
#    imp.update_all()
#    print(imp.data.head())
