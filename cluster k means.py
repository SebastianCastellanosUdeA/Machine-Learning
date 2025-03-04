# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 16:40:05 2025

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
df = pd.read_excel('consolidado prueba.xlsx')

# Cargar datos desde archivo GPKG
gdf = gpd.read_file('blz.gpkg')

cluster_labels = {0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5'}
cluster_colors = ['lightblue', 'darkblue', 'seagreen', 'lightgreen', 'lightpink']
# Preparar DataFrame para guardar resultados
results_df = pd.DataFrame()

def save_cluster_map(gdf, title, file_name):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    cmap = colors.ListedColormap(cluster_colors[:len(np.unique(gdf['cluster']))])
    gdf.plot(column='cluster_label', cmap=cmap, legend=True, ax=ax,
             legend_kwds={'title': 'Tipo de Clúster', 'loc': 'upper right'})
    ax.set_title(title)
    ax.set_axis_off()
    plt.savefig(f"{file_name}.png", bbox_inches='tight')
    plt.close(fig)
    
    
# Iterar sobre cada combinación de año y mes
for (year, month), group in df.groupby(['año', 'mes']):
    # Preprocesar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(group[['IVS', 'locais', 'PM']])
    data = gdf.merge(group, on='id')
    # Ejecutar K-Means
    kmeans = KMeans(n_clusters=5, init='k-means++', n_init=150, max_iter=1000, random_state=None)
    kmeans.fit(X_scaled)
    
    clusters = kmeans.fit_predict(X_scaled)
    data['cluster'] = clusters
    data['cluster_label'] = data['cluster'].apply(lambda x: cluster_labels[x])
    
    # Guardar resultados de centros y sumas de cuadrados
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    new_row = pd.DataFrame({
        'año': [year],
        'mes': [month],
        **{f'clustercenter{i+1}-{var}': [centers[i][j]] for i in range(5) for j, var in enumerate(['ivs', 'locais', 'pm'])},
        'sum_squares': [np.sum((X_scaled - np.mean(X_scaled, axis=0))**2)],
        'within_cluster_sum_sq': [np.sum((X_scaled - kmeans.cluster_centers_[kmeans.labels_])**2)],
        'between_cluster_sum_sq': [np.sum((X_scaled - np.mean(X_scaled, axis=0))**2) - np.sum((X_scaled - kmeans.cluster_centers_[kmeans.labels_])**2)],
        'ratio_sum_sq': [(np.sum((X_scaled - np.mean(X_scaled, axis=0))**2) - np.sum((X_scaled - kmeans.cluster_centers_[kmeans.labels_])**2)) / np.sum((X_scaled - np.mean(X_scaled, axis=0))**2)],
        **{f'within_cluster_ss_cluster{i+1}': [np.sum((X_scaled[kmeans.labels_ == i] - kmeans.cluster_centers_[i])**2)] for i in range(5)}
    })
    
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    save_cluster_map(data, f'Clusters for {year}-{month}', f'clusters_{year}_{month}')
    del data


# Guardar el DataFrame de resultados
results_df.to_excel('kmeans_results.xlsx', index=False)
