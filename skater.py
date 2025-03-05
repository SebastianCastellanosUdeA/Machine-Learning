import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.preprocessing import StandardScaler
from libpysal.weights import Rook
from spopt.region import Skater

# Cargar datos
df = pd.read_excel('consolidado prueba.xlsx')
gdf = gpd.read_file('blz.gpkg')

# Función para guardar mapas de clusters
def save_cluster_map(gdf, title, file_name):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    cluster_colors = ['lightblue', 'darkblue', 'seagreen', 'lightgreen', 'lightpink']
    cmap = colors.ListedColormap(cluster_colors[:len(gdf['cluster'].unique())])
    gdf.plot(column='cluster', cmap=cmap, legend=True, ax=ax)  
    ax.set_title(title)
    ax.set_axis_off()
    plt.savefig(f"{file_name}.png", bbox_inches='tight')
    plt.close(fig)

# DataFrame para guardar resultados
results_df = pd.DataFrame()

# Iterar por año y mes
for (year, month), group in df.groupby(['año', 'mes']):
    # Asegurar que los datos coincidan con el índice espacial
    group = group[group["id"].isin(gdf["id"])]

    # Escalar las variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(group[['IVS', 'locais', 'PM']])

    # Agregar las variables normalizadas al GeoDataFrame
    gdf_cluster = gdf[gdf["id"].isin(group["id"])].copy()
    gdf_cluster[['IVS', 'locais', 'PM']] = X_scaled

    # Construir matriz de pesos de Torre (Rook)
    w_rook = Rook.from_dataframe(gdf_cluster)

    # Aplicar SKATER con Torre (Rook Contiguity)
    skater = Skater(
        gdf=gdf_cluster,          # ✅ Pasamos el GeoDataFrame con las variables normalizadas
        w=w_rook,                 # ✅ Matriz de pesos espaciales
        attrs_name=['IVS', 'locais', 'PM'],  # ✅ Especificamos las columnas para clustering
        n_clusters=5
    )

    skater.solve()  # ✅ Ejecutamos el algoritmo sin necesidad de pasar datos aquí

    # Asignar etiquetas de cluster
    group['cluster'] = skater.labels_
    group['cluster_label'] = group['cluster'].apply(lambda x: f'Cluster {x + 1}')
    
    # Unir con datos espaciales
    month_gdf = gdf.merge(group[['id', 'cluster', 'cluster_label']], on='id')

    # Guardar mapa
    save_cluster_map(month_gdf, f'Clusters for {year}-{month}', f'clusters_{year}_{month}')
    
    # Guardar métricas de clusters
    within_cluster_vars = [np.var(X_scaled[np.array(skater.labels_) == i], axis=0).sum() for i in range(5)]
    new_row = pd.DataFrame({'año': [year], 'mes': [month], 'within_cluster_var': [within_cluster_vars]})

    
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# Guardar resultados en Excel
results_df.to_excel('skater_results.xlsx', index=False)
