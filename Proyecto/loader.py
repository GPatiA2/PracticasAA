import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pandas.io.parsers import read_csv


def cargarDatos() :
# Carga los datos desde los csv y devuelve un dataframe con las X, otro dataframe con las Y correspondientes y otro dataframe resultado de unir ambos
    d0 = read_csv('0.csv', header = None)
    d1 = read_csv('1.csv', header = None)
    d2 = read_csv('2.csv', header = None)
    d3 = read_csv('3.csv', header = None)

    df = pd.concat([d0,d1,d2,d3], axis = 0, ignore_index = True)

    return df.iloc[:,:-1] , df.iloc[:,-1] , df

def normalizar(df):
# Normaliza el dataframe que se pasa como parámetro y devuelve el scaler y el dataframe normalizado
    dfaux = df.copy()
 
    scaler = StandardScaler()
    scaler.fit(dfaux)
    res = scaler.transform(dfaux)
    
    dfres = pd.DataFrame(res)

    return dfres , scalers

def dividirDataSet(ds, rs):
# Divide el conjunto de entrenamiento en Entrenamiento, Validacion y Prueba
    EyV , P = train_test_split(ds, test_size = 0.2, random_state = rs)
    E ,   V = train_test_split(EyV, test_size = 0.25, random_state = rs)

    return E , V , P

def terminosPolinomicos(df,n):
# Inserta terminos polinomicos en el dataframe que se le pasa
    p = PolynomialFeatures(n)
    res = p.fit_transform(df)

    return pd.DataFrame(res)

