import numpy as np
import fun_basicas as fun
from scipy.optimize import minimize


class red_neuronal:

    ocultas = 0
    nodos_capa = []
    #matrices_pesos = [] # lo dejo aqui pero sino quito la inicialización no puede mejorar
    entradas = 0
    salidas = 0

    
    def __init__ (self , ocultas, nodos_capa, init, entradas, salidas):
        self.entradas = entradas
        self.salidas = salidas
        self.ocultas = ocultas
        self.nodos_capa = nodos_capa
        
        #M = np.random.random((nodos_capa[0], self.entradas + 1))  * (2 * init) - init
        #self.matrices_pesos.append(M)
        
        #for i in range(len(self.nodos_capa) - 1):
        #    M = np.random.random((nodos_capa[i+1], (1 + nodos_capa[i])))  * (2 * init) - init
        #    self.matrices_pesos.append(M)
            
        #M = np.random.random((salidas, (1 + nodos_capa[i + 1])))  * (2 * init) - init
        #self.matrices_pesos.append(M)
        
        
        

    def forward_prop(self, entrada,matrices_pesos):
        activaciones = []

        X = entrada
        n = X.shape[0]
        X = np.hstack([np.ones([n, 1]), X])
        activaciones.append(X)

        for i in range(len(matrices_pesos) - 1):

            A = fun.sigmoide(np.dot(X, matrices_pesos[i].T ))
            A = np.hstack([np.ones([n,1]), A])

            X = A

            activaciones.append(A)
            
        A = fun.sigmoide(np.dot(X, matrices_pesos[i + 1].T))
        activaciones.append(A)

        return activaciones

    
    
    
    def coste_reg(self,X,y,reg,matrices_pesos):
        activaciones = self.forward_prop(X,matrices_pesos)
        h = activaciones[-1]
        
        s1 = y * np.log(h)
        s2 = (1 - y) * np.log( 1 - h + 1e-6)

        c = (-1 / X.shape[0]) * np.sum(s1 + s2)
        e = sum([sum(sum(matrices_pesos[i][:,1:] ** 2)) for i in range(len(matrices_pesos))])

        return c + (reg / (2*X.shape[0])) * e

    
    
    
    def gradiente_reg(self, X, y, reg,matrices_pesos):
        # calculo de gradiente
        Deltas = [np.zeros(np.shape(matrices_pesos[i])) for i in range(len(matrices_pesos))]
        activaciones = self.forward_prop(X,matrices_pesos)

        for k in range(len(y)):
            activ = [activaciones[i][k, :] for i in range(len(activaciones))]

            ultimo = y[k, :]
            d = []
            
            j = (len(activaciones) - 1)
            
            daux = activ[j] - ultimo
            g_aux = activ[j-1] * ( 1 - activ[j-1] )
            Deltas[j - 1] = Deltas[j - 1] + np.dot(daux[:,np.newaxis], activ[j-1][np.newaxis, :])
            daux = np.dot(matrices_pesos[j-1].T, daux) * g_aux
            Deltas[j - 2] = Deltas[j - 2] + np.dot(daux[1:,np.newaxis], activ[j-2][np.newaxis, :])

            for j in range(len(activ)-3, 1, -1):
                g_aux = activ[j] * ( 1 - activ[j])
                daux = np.dot(matrices_pesos[j].T, daux[1:]) * g_aux
                Deltas[j - 1] = Deltas[j - 1] + np.dot(daux[1:,np.newaxis], activ[j-1][np.newaxis, :])
        
        # parte de regularización
        for i in range(len(Deltas)):
            Deltas[i] = Deltas[i] / len(y)
            Deltas[i][:, 1:] = Deltas[i][:, 1:] + (reg/len(y)) * matrices_pesos[i][:, 1:]

        return Deltas
    
    
    
    def init_aleatorio(self, init):
    
        # inicializar matrices aleatoriamente
        
        matrices_pesos = []
        
        M = np.random.random((self.nodos_capa[0], self.entradas + 1))  * (2 * init) - init
        matrices_pesos.append(M)
        
        i = -1
        
        for i in range(len(self.nodos_capa) - 1):
            M = np.random.random((self.nodos_capa[i+1], (1 + self.nodos_capa[i])))  * (2 * init) - init
            matrices_pesos.append(M)
            
        M = np.random.random((self.salidas, (1 + self.nodos_capa[i + 1])))  * (2 * init) - init
        matrices_pesos.append(M)
        
        pesos_ravel = np.concatenate(tuple(map(np.ravel, matrices_pesos)))
        
        return pesos_ravel
    
    
    
    def desenlazado(self, params_rn):
        """crea una lista con las matrices formadas con sus correctas dimensiones"""
        matrices_pesos = []
        
        # matriz desde la entrada hasta la primera capa oculta
        matrices_pesos.append(np.reshape(params_rn[:self.nodos_capa[0] * (self.entradas + 1)],
                                        (self.nodos_capa[0], (self.entradas + 1))))
        
        
        # las matriz entre capas ocultas
        ini = self.nodos_capa[0] * (self.entradas + 1)
        fin = ini
        
        for i in range(len(self.nodos_capa) - 1):
            fin = fin + (self.nodos_capa[i + 1] * (self.nodos_capa[i] + 1))
            matrices_pesos.append(np.reshape(params_rn[ini : fin], (self.nodos_capa[i + 1], (self.nodos_capa[i] + 1))))
            ini = fin
            
            
        # la matriz desde la ultima oculta hasta la salida    
        matrices_pesos.append(np.reshape(params_rn[ini :], (self.salidas, (self.nodos_capa[len(self.nodos_capa) - 1] + 1))))
        
        return matrices_pesos
    
    
    

    def backprop(self, params_rn, X, y, reg):
        
        matrices_pesos = self.desenlazado(params_rn)
        
        deltas = self.gradiente_reg(X,y,reg, matrices_pesos)
        coste = self.coste_reg(X,y,reg, matrices_pesos)

        gr = tuple(map(np.ravel, deltas))

        gradiente = np.concatenate(gr)

        return coste, gradiente

    
    
    
    def entrenar(self, X, y, Vx, Vy, Px, Py, reg, iters, init):
        
        pesos_ravel = self.init_aleatorio(init)
        
        # calculo del minimo

        fmin = minimize(fun = self.backprop , x0 = pesos_ravel , args = (X,y,reg),
                        method = 'TNC' , jac = True , options = {'maxiter' : iters})
        
        matrices_pesos = self.desenlazado(fmin.x)

        p1 = self.prueba_neurona(Vx, Vy, matrices_pesos)
        print("validación = {}".format(p1))
        p2 = self.prueba_neurona(Px, Py, matrices_pesos)
        print("prueba = {}".format(p2))
        
        
        
    def prueba_neurona(self, X, y, matrices_pesos):
        """función que devuelve el porcentaje de acierto de una red neuronal utilizando unas matrices de pesos dadas"""
        n = len(y)

        y = np.ravel(y)

        forward = self.forward_prop(X, matrices_pesos)
        
        result = forward[len(forward)-1]

        result = np.argmax(result, axis=1)

        return (sum((result + 1)%4 == y) / n * 100)