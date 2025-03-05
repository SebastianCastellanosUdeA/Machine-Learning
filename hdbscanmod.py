import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as colors

# Cargar datos desde Excel
df = pd.read_excel('consolidado prueba.xlsx')

# Cargar datos geográficos desde archivo GPKG
gdf = gpd.read_file('blz.gpkg')

def save_cluster_map(gdf, title, file_name):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    cluster_colors = ['lightblue', 'darkblue', 'seagreen', 'lightgreen', 'lightpink', 'red', 'orange', 'purple', 'brown']
    unique_clusters = gdf['cluster_label'].unique()
    cmap = colors.ListedColormap(cluster_colors[:len(unique_clusters)])
    gdf.plot(column='cluster_label', cmap=cmap, legend=True, ax=ax,
             legend_kwds={'title': 'Tipo de Clúster', 'loc': 'upper right'})
    ax.set_title(title)
    ax.set_axis_off()
    plt.savefig(f"{file_name}.png", bbox_inches='tight')
    plt.close(fig)

# Preparar DataFrame para guardar resultados
results_df = pd.DataFrame()

# Iterar sobre cada combinación de año y mes
for (year, month), group in df.groupby(['año', 'mes']):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(group[['IVS', 'locais', 'PM']])
    
    # Ejecutar HDBSCAN de la manera tradicional
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    clusterer.fit(X_scaled)
    
    # Asignar etiquetas de cluster
    group['cluster'] = clusterer.labels_
    group['cluster_label'] = group['cluster'].apply(lambda x: f'Cluster {x+1}' if x != -1 else 'Ruido')
    
    # Fusionar con datos geográficos
    month_gdf = gdf.merge(group[['id', 'cluster', 'cluster_label']], on='id', how='left')
    save_cluster_map(month_gdf, f'Clusters for {year}-{month}', f'clusters_{year}_{month}')
    
    # Guardar resultados de clúster
    unique_clusters = np.unique(clusterer.labels_)
    cluster_centers = [np.mean(X_scaled[clusterer.labels_ == i], axis=0) if i != -1 else [np.nan, np.nan, np.nan] for i in unique_clusters]
    centers = scaler.inverse_transform(cluster_centers)
    
    new_row = pd.DataFrame({
        'año': [year],
        'mes': [month],
        **{f'clustercenter{i+1}-{var}': [centers[i][j]] for i in range(len(unique_clusters)) for j, var in enumerate(['ivs', 'locais', 'pm'])},
        'num_clusters': [len(unique_clusters)-1],
        'num_ruido': [np.sum(clusterer.labels_ == -1)],
    })
    
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# Guardar el DataFrame de resultados
results_df.to_excel('hdbscan_results.xlsx', index=False)
