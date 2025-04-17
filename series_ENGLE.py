# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:58:54 2025

@author: nikit
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Supongamos que tu dataframe se llama df y la columna con los datos es 'datos'
# Realizamos la prueba ADF

def test_stationarity(series):
    result = adfuller(series)
    print('Estadístico ADF:', result[0])
    print('Valor p:', result[1])
    print('Número de lags usados:', result[2])
    print('Número de observaciones usadas para la prueba:', result[3])
    print('Valores críticos:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    # Si el valor p es menor que 0.05, podemos decir que la serie es estacionaria
    if result[1] < 0.05:
        print("La serie es estacionaria.")
    else:
        print("La serie NO es estacionaria.")

df = pd.read_excel('CENTRO.xlsx')

# Aplicamos la prueba ADF a la columna 'datos'
test_stationarity(df['PM'])


# Supongamos que tu DataFrame se llama df y tiene las columnas 'IVS' y 'TUC'

# Estimación de la regresión entre IVS y TUC
X = df['TUC']
X = sm.add_constant(X)  # Añadimos el intercepto
y = df['IVS']

model = sm.OLS(y, X).fit()  # Regresión lineal
residuos = model.resid  # Residuos de la regresión

# Aplicar la prueba ADF sobre los residuos
def test_stationarity(series):
    result = adfuller(series)
    print('Estadístico ADF:', result[0])
    print('Valor p:', result[1])
    print('Número de lags usados:', result[2])
    print('Número de observaciones usadas para la prueba:', result[3])
    print('Valores críticos:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    # Si el valor p es menor que 0.05, podemos decir que la serie es estacionaria
    if result[1] < 0.05:
        print("Los residuos son estacionarios. Las series están cointegradas.")
    else:
        print("Los residuos NO son estacionarios. Las series NO están cointegradas.")

# Aplicar la prueba ADF sobre los residuos de la regresión
test_stationarity(residuos)



from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Seleccionamos las columnas que vamos a analizar (IVS y TUC)
df_coint = df[['IVS', 'TUC']]

# Realizamos el test de cointegración de Johansen
johansen_test = coint_johansen(df_coint, det_order=0, k_ar_diff=10)  # Elige 12 como número de rezagos (ajustar según el contexto)

# Resultados del test de cointegración
print("Estadísticas de la prueba de traza (trace test):")
print(johansen_test.lr1)  # Estadísticas de la prueba de traza
print("Valores críticos para la prueba de traza:")
print(johansen_test.cvt)  # Valores críticos para comparar

# Si la estadística de la prueba de traza es mayor que el valor crítico, podemos rechazar la hipótesis nula

