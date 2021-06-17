import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as skl
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

def evalua_Lambdas(Ex, Ey, Vx, Vy, Px, Py, params):
    
    scores = np.zeros((len(params), len(params)))
    res = list()

    for v in params:
    
        for sigma in params:
            svm = skl.SVC(kernel = 'rbf', C = v, gamma = 1/(2*sigma**2))
            svm.fit(Ex, Ey)
            xpred = svm.predict(Vx)
            acertados = sum(xpred == Vy) / xpred.shape[0] * 100
            res.append(acertados)
            scores[params.index(v), params.index(sigma)] = acertados
        
    print(scores)
    ind = np.where(scores == scores.max())

    indp1 = ind[0][0]
    indp2 = ind[1][0]


    svm = skl.SVC(kernel = 'rbf', C = params[indp1], gamma = 1/(2*params[indp2]**2))
    svm.fit(Ex, Ey)
    print(accuracy_score(Py, svm.predict(Px)))

    print("C = {} , sigma = {}".format(params[indp1],params[indp2]))
    
    return params[indp1], params[indp2]


def evalua_Kernels(Ex, Ey, Vx, Vy, Px, Py, C, sigma):
    kernels= ['rbf', 'poly', 'sigmoid']
    scores = np.zeros((len(kernels)))
    res = list()

    for k in kernels:
        svm = skl.SVC(kernel = k, C = C, gamma = 1/(2*sigma**2))
        svm.fit(Ex, Ey)
        xpred = svm.predict(Vx)
        acertados = sum(xpred == Vy) / xpred.shape[0] * 100
        print(k, "acertados : ", acertados)
        res.append(acertados)
        scores[kernels.index(k)] = acertados
        
    print(scores)
    ind = np.where(scores == scores.max())

    svm = skl.SVC(kernel = kernels[ind[0][0]] , C = C , gamma = 1/(2*sigma**2))
    svm.fit(Ex, Ey)
    print(kernels[ind[0][0]], accuracy_score(Py, svm.predict(Px)))
    
    return kernels[ind[0][0]], accuracy_score(Py, svm.predict(Px))