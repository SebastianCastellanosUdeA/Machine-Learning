# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:54:50 2025

@author: sebas
"""

import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

# Cargar datos desde Excel
df_excel = pd.read_excel('prueba hexagonos.xlsx')

# Cargar datos desde archivo GPKG
gdf = gpd.read_file('blz.gpkg')

# Unir los datos usando la columna 'id'
data = gdf.merge(df_excel, on='id')

# Seleccionar las columnas de interés
columns = ['PM', 'IVS', 'locais']
X = data[columns]

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Configurar y correr el modelo K-Means
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=150, max_iter=1000, random_state=None)
clusters = kmeans.fit_predict(X_scaled)

# Añadir los resultados al GeoDataFrame
data['cluster'] = clusters

cluster_labels = {0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5'}
cluster_colors = ['lightblue', 'darkblue', 'seagreen', 'lightgreen', 'lightpink']  # Lista de colores
data['cluster_label'] = data['cluster'].apply(lambda x: cluster_labels[x])

def save_cluster_map(gdf, title, file_name):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    cmap = colors.ListedColormap(cluster_colors[:len(np.unique(gdf['cluster']))])
    gdf.plot(column='cluster_label', cmap=cmap, legend=True, ax=ax,
             legend_kwds={'title': 'Tipo de Clúster', 'loc': 'upper right'})
    ax.set_title(title)
    ax.set_axis_off()
    plt.savefig(f"{file_name}.png", bbox_inches='tight')
    plt.close(fig)

# Guardar el mapa de clusters
save_cluster_map(data, 'Mapa de Clusters K-Means2', 'clusters_kmeans2')


# Imprimir las métricas y los centros de cada cluster
print("Centros de clusters:")
print(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=columns))

# Calcular sumas de cuadrados
total_ss = np.sum((X_scaled - np.mean(X_scaled, axis=0))**2)
within_ss = np.sum((X_scaled - kmeans.cluster_centers_[kmeans.labels_])**2)
between_ss = total_ss - within_ss
ratio = between_ss / total_ss

print(f"Total sum of squares: {total_ss:.4f}")
print(f"Within-cluster sum of squares: {within_ss:.4f}")
print(f"Between-cluster sum of squares: {between_ss:.4f}")
print(f"Ratio of between to total sum of squares: {ratio:.4f}")

# Calculamos la suma de cuadrados dentro de cada cluster
within_cluster_ss = np.zeros(kmeans.n_clusters)

for i, center in enumerate(kmeans.cluster_centers_):
    mask = kmeans.labels_ == i
    within_cluster_ss[i] = np.sum((X_scaled[mask] - center)**2)

# Imprimir la suma de cuadrados dentro de cada cluster
for i, ss in enumerate(within_cluster_ss, 1):
    print(f"C{i}|{ss:.4f}")