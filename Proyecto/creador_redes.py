import numpy as np
import fun_basicas as fun
from scipy.optimize import minimize

class red_neuronal:

    ocultas = 0
    nodos_capa = []
    matrices_pesos = [] 
    entradas = 0
    salidas = 0

    def __init__ (self , ocultas, nodos_capa, init, entradas, salidas):
        self.entradas = entradas
        self.salidas = salidas
        self.ocultas = ocultas
        self.nodos_capa = nodos_capa
        for i in range(len(self.nodos_capa)):
            M = np.random.random([nodos_capa[i+1], 1 + nodos_capa[i]])  * (2 * init) - init
            matrices_pesos.append(M)

    def foward_prop(self, entrada):
        activaciones = []

        X = entrada
        n = X.shape[0]
        X = np.hstack([np.ones[n,1], X])
        activaciones.append(X)
        

        for i in range(len(matrices_pesos)):

            A = fun.sigmoide(np.dot(matrices_pesos[i].T , X))
            A = np.hstack([np.ones([n,1]), A])

            X = A

            activaciones.append(A)

        return activaciones

    def coste_reg(self,X,y,reg):
        activaciones = self.forward_prop(X)
        h = activaciones[-1]
        
        s1 = y * np.log(h)
        s2 = (1-y) * np.log( 1 - h * 1e-6)

        c = (-1 / X.shape[0]) * np.sum(s1 + s2)
        e = sum([sum(matrices_pesos[i][:,1:] ** 2) for i in range(len(matrices_pesos))])

        return c + (reg / (2*X.shape[0])) * e

    def gradiente_reg(self, X, y, reg):

        Deltas = [np.zeros(np.shape(self.matrices_pesos[i])) for i in range(len(self.matrices_pesos))]
        activaciones = self.foward_prop(self, X)

        for k in range(len(y)):
            activ = [activaciones[i][k, :] for i in range(len(activaciones))]

            ultimo = y[k, :]
            d = []

            for j in range(len(activ)-1, 1, -1):
                daux = activ[j] - ultimo
                g_aux = activ[j-1] * ( 1 - activ[j-1] )
                Deltas[j] = Deltas[j] + np.dot(daux[1:,np.newaxis], activ[j][np.newaxis, :])
        
        for i in range(len(Deltas)):
            Deltas[i] = Deltas[i] / len(y)
            Deltas[i][:, 1:] = Deltas[i][:, 1:] + reg/len(y) * matrices_pesos[:, 1:]

        return Deltas

    def backprop(self, X, y, reg):
        
        coste = coste_reg(X,y,reg)
        deltas = gradiente_reg(X,y,reg)

        gr = tuple(map(np.ravel, deltas))

        gradiente = np.concatenate(gr)

        return coste, gradiente

    def entrenar(self, X, y, reg, iters):
        
        pesos_ravel = tuple(map(np.ravel,self.matrices_pesos))

        fmin = minimize(fun = self.backprop , x0 = pesos_ravel , args = (X,y,reg),
                        method = 'TNC' , jac = True , options = {'maxiter' : iters})


        

    






