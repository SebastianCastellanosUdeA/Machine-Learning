# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:26:04 2025

@author: sebas
"""

import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen, Rook
from esda.moran import Moran, Moran_BV, Moran_Local, Moran_Local_BV
import itertools

# Cargar los datos
let = 'consolidado'
df = pd.read_excel('consolidado.xlsx')
gdf = gpd.read_file('blz.gpkg')

# Convertir ID en gdf a entero
gdf['id'] = gdf['id'].astype(float).astype(int)

# Convertir ID en df a entero
df['id'] = df['id'].astype(int)

# Unir los datos
data = gdf.merge(df, on='id')

# Establecer ID como índice
data.set_index('id', inplace=True)

# Crear pesos espaciales
w = Rook.from_dataframe(gdf)
w.transform = 'r'

# Variables a analizar
variables = ['PM', 'locais', 'IVS']



# Bucle para calcular Moran's I, Moran Local, y Moran_BV por año y mes

lisa_counts = []
for year in data['año'].unique():
    for month in data['mes'].unique():
        temp_data = data[(data['año'] == year) & (data['mes'] == month)]
        if not temp_data.empty:
            if (len(temp_data) >= len(gdf)):
                for var in variables:
                    ml = Moran_Local(temp_data[var], w)
                    significance = ml.p_sim < 0.05
                    # Sumarización de clusters LISA
                    labels = ['No significativo' if not sig else
                              'Alto-Alto' if q == 1 else
                              'Bajo-Bajo' if q == 2 else
                              'Bajo-Alto' if q == 3 else
                              'Alto-Bajo' for q, sig in zip(ml.q, significance)]


                    label_counts = pd.Series(labels).value_counts().reindex(['Alto-Alto', 'Bajo-Bajo', 'Bajo-Alto', 'Alto-Bajo', 'No significativo'], fill_value=0)
                    lisa_counts.append((let, year, month, var, label_counts['Alto-Alto'], label_counts['Bajo-Bajo'], label_counts['Bajo-Alto'], label_counts['Alto-Bajo'], label_counts['No significativo']))
                
                for var1, var2 in itertools.combinations(variables, 2):
                    ml_bv = Moran_Local_BV(temp_data[var1], temp_data[var2], w)
                    significance2 = ml_bv.p_sim < 0.05
                    labels_bv = ['No significativo' if not sig else
                                 'Alto-Alto' if q == 1 else
                                 'Bajo-Bajo' if q == 2 else
                                 'Bajo-Alto' if q == 3 else
                                 'Alto-Bajo' for q, sig in zip(ml_bv.q, significance2)]

                    label_counts_bv = pd.Series(labels_bv).value_counts().reindex(['Alto-Alto', 'Bajo-Bajo', 'Bajo-Alto', 'Alto-Bajo', 'No significativo'], fill_value=0)
                    lisa_counts.append((let, year, month, f'{var1}-{var2}', label_counts_bv['Alto-Alto'], label_counts_bv['Bajo-Bajo'], label_counts_bv['Bajo-Alto'], label_counts_bv['Alto-Bajo'], label_counts_bv['No significativo']))
                    
                    ml_bv2 = Moran_Local_BV(temp_data[var2], temp_data[var1], w)
                    significance3 = ml_bv2.p_sim < 0.05
                    labels_bv2 = ['No significativo' if not sig else
                                 'Alto-Alto' if q == 1 else
                                 'Bajo-Bajo' if q == 2 else
                                 'Bajo-Alto' if q == 3 else
                                 'Alto-Bajo' for q, sig in zip(ml_bv2.q, significance3)]

                    label_counts_bv2 = pd.Series(labels_bv2).value_counts().reindex(['Alto-Alto', 'Bajo-Bajo', 'Bajo-Alto', 'Alto-Bajo', 'No significativo'], fill_value=0)
                    lisa_counts.append((let, year, month, f'{var2}-{var1}', label_counts_bv2['Alto-Alto'], label_counts_bv2['Bajo-Bajo'], label_counts_bv2['Bajo-Alto'], label_counts_bv2['Alto-Bajo'], label_counts_bv2['No significativo']))


# Convertir los resultados a DataFrame para mejor visualización

lisa_summary_df = pd.DataFrame(lisa_counts, columns=['SER','Año', 'Mes', 'Variable', 'Alto-Alto', 'Bajo-Bajo', 'Bajo-Alto', 'Alto-Bajo', 'No significativo'])

# Guardar
lisa_summary_df.to_excel('lisa_consolidado.xlsx', index=False)






