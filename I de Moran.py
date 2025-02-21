# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:07:01 2025

@author: sebas
"""

import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen, Rook
from esda.moran import Moran, Moran_BV
import itertools

# Cargar los datos
df = pd.read_excel('SER12.xlsx')
gdf = gpd.read_file('hexagonos_SER 12.gpkg')


# Convertir ID en gdf a entero (quitar decimales primero si es necesario)
gdf['id'] = gdf['id_1'].astype(float).astype(int)

# Convertir ID en df a entero
df['id'] = df['id'].astype(int)

# Unir los datos
data = gdf.merge(df, on='id')

# Asumiendo que gdf contiene solo los polígonos de los 121 barrios únicos
#w = Queen.from_dataframe(gdf)
w = Rook.from_dataframe(gdf)
w.transform = 'r'

# Suponiendo que 'NI' o 'NID2' es el identificador de los barrios
data.set_index('id', inplace=True)  # Asegura que el índice sea el identificador del barrio


# Variables a analizar
variables = ['PM', 'locais', 'IVS']

# Bucle para calcular Moran's I por año y mes
results = []
for year in data['año'].unique():
    for month in data['mes'].unique():
        temp_data = data[(data['año'] == year) & (data['mes'] == month)]
        if not temp_data.empty:
            if (len(temp_data)>=5):
                # Cálculo univariado
                for var in variables:
                    mi = Moran(temp_data[var], w)
                    results.append((year, month, var, 'Univariado', mi.I, mi.p_sim))
                
                # Cálculo bivariado
                for var1, var2 in itertools.combinations(variables, 2):
                    mi_bv = Moran_BV(temp_data[var1], temp_data[var2], w)
                    results.append((year, month, f'{var1}-{var2}', 'Bivariado', mi_bv.I, mi_bv.p_sim))
                    mi_bv = Moran_BV(temp_data[var2], temp_data[var1], w)
                    results.append((year, month, f'{var2}-{var1}', 'Bivariado', mi_bv.I, mi_bv.p_sim))

# Convertir los resultados a DataFrame para mejor visualización
results_df = pd.DataFrame(results, columns=['Año', 'Mes', 'Variables', 'Tipo', 'I de Moran', 'P-valor'])

# Guardar 
results_df.to_excel('i_moran_ser_12.xlsx', index=False)

# Mostrar resultados
#print(results_df)


