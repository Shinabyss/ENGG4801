'''
Created on 17 Oct. 2017

@author: user
'''
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

Xtrain = pd.read_csv('XtrainKNNCat.csv').values
Xtest = pd.read_csv('XtestKNNCat.csv').values

'''
pca = PCA()
pca.fit(Xtrain)

Xtrain_pca = pca.transform(Xtrain)
Xtest_pca = pca.transform(Xtest)

np.savetxt("pcaTrain.csv", Xtrain_pca, delimiter=",")
np.savetxt("pcaTest.csv", Xtest_pca, delimiter=",")
'''

pcar = PCA(n_components=30)
pcar.fit(Xtrain)

Xtrain_pcar = pcar.transform(Xtrain)
Xtest_pcar = pcar.transform(Xtest)

np.savetxt("pcaReducedTrain.csv", Xtrain_pcar, delimiter=",")
np.savetxt("pcaReducedTest.csv", Xtest_pcar, delimiter=",")