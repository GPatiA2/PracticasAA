import numpy as np

def sigmoide(Z):
    sigmoide = 1 / (1 + np.exp(-Z))
    return sigmoide

def preparaFunciones(Lambda):
    c = lambda Theta, X, Y: coste_reg(Theta, X, Y, Lambda)
    gr = lambda Theta, X, Y: gradiente_reg(Theta, X, Y, Lambda)

    return (c, gr)


def prepara_datos(X, y, et):
    Y2 = (y == et) * 1
    ## Aquí hay que hacer ravel de Y2 para pasar de (5000,1) a (5000,1)
    ## y conseguir que funcione como en la practica anterior
    Y2 = np.ravel(Y2)
    return (X, Y2)


def one_hot(y, et):
    # Transforma y en one_hot con et numero de etiquetas
    m = len(y)

    y = (y - 1)
    y_onehot = np.zeros((m, et))

    for i in range(m):
        y_onehot[i][y[i]] = 1

    return y_onehot