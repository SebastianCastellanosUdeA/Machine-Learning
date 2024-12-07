import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from shapely.geometry import Polygon
import networkx as nx

# Función para crear una matriz de adyacencia tipo "torre"
def create_adjacency_matrix(gdf):
    G = nx.Graph()
    for idx, poly in enumerate(gdf.geometry):
        G.add_node(idx)
        for jdx, other_poly in enumerate(gdf.geometry):
            if idx != jdx and poly.touches(other_poly) and not poly.crosses(other_poly):
                G.add_edge(idx, jdx)
    return nx.adjacency_matrix(G)

# Función para imputar valores manualmente
def imputar_knn_manual(valores, matriz_distancias, k):
    valores_imputados = valores.copy()
    for i in range(len(valores)):
        if np.isnan(valores[i]):
            # Identificar vecinos más cercanos
            distancias = matriz_distancias[i, :]
            vecinos_idx = np.argsort(distancias)
            vecinos_idx = vecinos_idx[~np.isnan(valores[vecinos_idx])][:k]  # Solo vecinos con valores
            if len(vecinos_idx) > 0:
                valores_imputados[i] = np.mean(valores[vecinos_idx])  # Promedio de los vecinos
    return valores_imputados

# Carga de datos
barrios_gpkg = "BairrosFortal.gpkg"  # Reemplaza con la ruta real
datos_excel = "idh.xlsx"   # Reemplaza con la ruta real

# Leer datos
barrios = gpd.read_file(barrios_gpkg)
datos = pd.read_excel(datos_excel)

# Unir datos por el nombre del barrio
barrios = barrios.merge(datos, on="Nid2", how="left")

# Crear matriz de adyacencia tipo torre
adjacency_matrix = create_adjacency_matrix(barrios)
# Convertir la matriz de adyacencia a tipo float y ajustar
matriz_distancias = adjacency_matrix.toarray().astype(float)
matriz_distancias[matriz_distancias == 0] = np.inf  # Donde no hay conexión es infinito
np.fill_diagonal(matriz_distancias, 0)  # La diagonal es 0 porque son los mismos barrios


# Imputar datos año por año
resultados = []
columnas_a_imputar = ["IDH_2020_O", "IDH_2010_O", "IDH_2000_O",
                      "h2000","h2001","h2002","h2003","h2004",
                      "h2005","h2006","h2007","h2008",
                      "h2009","h2010","h2011","h2012",
                      "h2013","h2014","h2015","h2016",
                      "h2017","h2018","h2019","h2020",
                      "h2021","h2022","h2023"]
for col in columnas_a_imputar:
    print(f"Procesando columna: {col}")
    
    # Crear dataset para imputación
    valores = barrios[col].values
    valores_originales = valores.copy()
    
    # Dividir en conjunto de entrenamiento y validación
    indices_no_nulos = ~np.isnan(valores)
    X_train, X_val, y_train, y_val = train_test_split(
        np.where(indices_no_nulos)[0],
        valores[indices_no_nulos],
        test_size=0.2,
        random_state=42
    )

    # Marcar valores del conjunto de validación como NaN
    valores[X_val] = np.nan

    mejores_k = None
    mejor_error = float("inf")
    mejor_imputacion = None

    # Probar diferentes valores de k
    for k in range(1, 5):
        print(f"Probando k={k}")
        
        # Imputar valores usando los k vecinos más cercanos
        valores_imputados = imputar_knn_manual(valores, matriz_distancias, k)

        # Calcular error
        y_val_pred = valores_imputados[X_val]
        error = mean_squared_error(y_val, y_val_pred)

        if error < mejor_error:
            mejor_error = error
            mejores_k = k
            mejor_imputacion = valores_imputados

    print(f"Mejor k para {col}: {mejores_k} con error {mejor_error}")
    
    # Guardar resultados imputados para esta columna
    barrios[col] = mejor_imputacion

# Guardar los resultados en el archivo Excel original
barrios.to_excel("datos_imputados.xlsx", index=False)
