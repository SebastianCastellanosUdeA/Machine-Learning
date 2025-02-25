# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:43:22 2025

@author: sebas
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from libpysal.weights import Queen, Rook
from esda.moran import Moran, Moran_BV, Moran_Local, Moran_Local_BV
import itertools
import numpy as np
from matplotlib import colors

# Cargar los datos
df = pd.read_excel('consolidado.xlsx')
gdf = gpd.read_file('blz.gpkg')
gdf['id'] = gdf['id'].astype(float).astype(int)
df['id'] = df['id'].astype(int)
data = gdf.merge(df, on='id')
data.set_index('id', inplace=True)
w = Rook.from_dataframe(gdf)
w.transform = 'r'
variables = ['PM', 'locais', 'IVS']



# Función para guardar el mapa como imagen
def save_lisa_map(gdf, ml, title, file_name):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Sumarización de clusters LISA con asignación de colores
    significance = ml.p_sim < 0.05
    labels = np.array(['No significativo' if not sig else
                      'Alto-Alto' if q == 1 else
                      'Bajo-Bajo' if q == 3 else
                      'Bajo-Alto' if q == 2 else
                      'Alto-Bajo' for q, sig in zip(ml.q, significance)])
    color_map = {'Alto-Alto': 'red', 'Bajo-Bajo': 'blue', 'Bajo-Alto': 'lightblue', 'Alto-Bajo': 'lightcoral', 'No significativo': '#f0f0f0'}
    colors_list = [color_map[label] for label in labels]

    gdf['label'] = labels  # Asignar etiquetas al GeoDataFrame

    # Crear mapa de colores personalizado
    cmap = colors.ListedColormap([color_map[label] for label in np.unique(labels)])
    norm = colors.Normalize(vmin=0, vmax=len(np.unique(labels)))

    # Asignación del mapa de colores al plot
    gdf.plot(column='label', cmap=cmap, norm=norm, legend=True, ax=ax,
             legend_kwds={'title': 'Tipo de Clúster'})

    ax.set_title(title)
    ax.set_axis_off()
    plt.savefig(f"{file_name}.png", bbox_inches='tight')
    plt.close(fig)
    
    


# Bucle para calcular y guardar Moran's I, Moran Local, y Moran_BV por año y mes
for year in data['año'].unique():
    for month in data['mes'].unique():
        temp_data = data[(data['año'] == year) & (data['mes'] == month)]
        if not temp_data.empty and (len(temp_data) >= len(gdf)):
            for var in variables:
                ml = Moran_Local(temp_data[var], w)
                save_lisa_map(gdf, ml, f"LISA Cluster Map of {var} - {year}/{month}", f"{var}_{year}_{month}_univar")

            for var1, var2 in itertools.combinations(variables, 2):
                ml_bv = Moran_Local_BV(temp_data[var1], temp_data[var2], w)
                save_lisa_map(gdf, ml_bv, f"LISA BV Cluster Map of {var1} vs {var2} - {year}/{month}", f"{var1}_{var2}_{year}_{month}_bivar")
                ml_bv2 = Moran_Local_BV(temp_data[var2], temp_data[var1], w)
                save_lisa_map(gdf, ml_bv2, f"LISA BV Cluster Map of {var2} vs {var1} - {year}/{month}", f"{var2}_{var1}_{year}_{month}_bivar")
