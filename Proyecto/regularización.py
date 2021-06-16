from loader import *
from fun_basicas import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

def coste(Theta, X, Y):
    G = sigmoide(np.dot(X, Theta))
    sum1 = np.dot(Y, np.log(G))
    sum2 = np.dot((1-Y), np.log(1 - G))
    return (-1 / X.shape[0]) * (sum1 + sum2)

def gradiente(Theta, X, Y):
    m = X.shape[0]
    G = sigmoide( np.matmul(X,Theta) )
    gradiente  = (1 / len(Y)) * np.matmul(X.T, G - Y)
    return gradiente


def coste_reg(Theta, X, Y, Lambda):
    c = coste(Theta, X, Y)
    m = X.shape[0]
    e = 0

    for t in range(1, len(Theta)):
        e += Theta[t] ** 2

    return c + (Lambda / (2 * m)) * e

def gradiente_reg(Theta,X,Y,Lambda):
    m = X.shape[0]
    gr = gradiente(Theta,X,Y)
    theta2 = (Lambda/m)*Theta
    return (gr + theta2)


def optimiza_reg(X, Y, Lambda, et):
    X, Y = prepara_datos(X, Y, et)
    c, gr = preparaFunciones(Lambda)

    T = np.zeros(X.shape[1])

    result = opt.fmin_tnc(func=c, x0=T, fprime=gr, args=(X, Y))
    c_f = coste(result[0], X, Y)
    print("coste:", c_f)
    return result[0]


def oneVsAll(X, y, num_etiquetas, reg):
    params = []

    # Por cada tipo de etiqueta se devuelve la Theta optima que reconoce la misma y se añade a un array de Thetas
    for et in range(0, num_etiquetas):
        p = optimiza_reg(X, y, reg, et)
        params.append(p)
    return np.array(params)


def evalua():
    X, Y, _ = cargarDatos()
    X = X.to_numpy()
    Y = Y.to_numpy()

    L = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    for Lamda in L:
        Theta = oneVsAll(X, Y, 4, Lamda)
        asig = []
        for i in range(X.shape[0]):
            l = np.dot(Theta, X[i])
            m = max(l)
            i = np.where(l == m)
            asig.append(i[0][0])

        y2 = np.ravel(Y)
        t = (asig == y2) * 1
        perc = (sum(t) / 5000) * 100
        print("Porcentaje de aciertos con lambda = {} : ".format(Lamda), perc, "%")

