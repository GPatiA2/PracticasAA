from loader import *
from fun_basicas import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


def coste(theta1, theta2, X, Y, num_etiquetas):  # Y preparada

    A1, A2, h = forward_prop(X, theta1, theta2)

    sum1 = Y * np.log(h)
    sum2 = (1 - Y) * np.log(1 - h + 1e-6)

    return (-1 / X.shape[0]) * np.sum(sum1 + sum2)


def coste_reg(theta1, theta2, X, Y, num_etiquetas, Lambda):
    c = coste(theta1, theta2, X, Y, num_etiquetas)
    m = X.shape[0]

    e = sum(sum(theta1[:, 1:] ** 2)) + sum(sum(theta2[:, 1:] ** 2))

    return c + (Lambda / (2 * m)) * e


def forward_prop(X, theta1, theta2):
    n = X.shape[0]

    # Se añade una fila de unos a la matriz inicial
    X = np.hstack([np.ones([n, 1]), X])

    # La capa oculta utiliza la primera matriz de pesos para crear sus neuronas y le añade una fila de unos
    Oculta = sigmoide(np.dot(X, theta1.T))
    Oculta = np.hstack([np.ones([n, 1]), Oculta])

    # El resultado se calcula pasando por la segunda matriz de pesos todas las neuronas de la capa oculta
    Resultado = sigmoide(np.dot(Oculta, theta2.T))

    return X, Oculta, Resultado


def gradiente(theta1, theta2, X, y):
    # Creamos los Delta con la forma de theta pero inicializados a cero
    Delta1 = np.zeros(np.shape(theta1))
    Delta2 = np.zeros(np.shape(theta2))

    m = len(y)

    # Se realiza la propagación hacia delante
    A1, A2, h = forward_prop(X, theta1, theta2)

    # Se realiza la propagación hacia atras para cada
    # elemento para comprobar el fallo
    for k in range(m):
        a1k = A1[k, :]
        a2k = A2[k, :]
        a3k = h[k, :]
        yk = y[k, :]

        d3 = a3k - yk
        g_prima = (a2k * (1 - a2k))
        d2 = np.dot(theta2.T, d3) * g_prima

        Delta1 = Delta1 + np.dot(d2[1:, np.newaxis], a1k[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3[:, np.newaxis], a2k[np.newaxis, :])

    # Se devuelven los Deltas que corresponden al gradiente
    return Delta1 / m, Delta2 / m


def gradiente_reg(theta1, theta2, X, y, Lambda):
    m = len(y)
    Delta1, Delta2 = gradiente(theta1, theta2, X, y)

    # A cada elemento del gradiente (menos la primera columna) se le añade el termino de regularización Lambda
    # multiplicado por cada elemento de las matriz theta 1 y theta2
    Delta1[:, 1:] = Delta1[:, 1:] + (Lambda / m) * theta1[:, 1:]
    Delta2[:, 1:] = Delta2[:, 1:] + (Lambda / m) * theta2[:, 1:]

    return Delta1, Delta2


def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    # backprop devuelve una tupla (coste, gradiente) con el coste y el gradiente de
    # una red neuronal de tres capas , con num_entradas , num_ocultas nodos en la capa
    # oculta y num_etiquetas nodos en la capa de salida. Si m es el numero de ejemplos
    # de entrenamiento, la dimensión de ’X’ es (m, num_entradas) y la de ’y’ es
    # (m, num_etiquetas)

    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    m = len(y)

    D1, D2 = gradiente_reg(theta1, theta2, X, y, reg)

    coste = coste_reg(theta1, theta2, X, y, num_etiquetas, reg)

    gradiente = np.concatenate((np.ravel(D1), np.ravel(D2)))

    return coste, gradiente


def prueba_neurona(X, y, theta1, theta2):
    """función que devuelve el porcentaje de acierto de una red neuronal utilizando unas matrices de pesos dadas"""
    n = len(y)

    y = np.ravel(y)

    _, _, result = forward_prop(X, theta1, theta2)

    result = np.argmax(result, axis=1)

    return (sum(result + 1 == y) / n * 100)