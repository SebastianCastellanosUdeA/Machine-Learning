# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 18:04:26 2025

@author: sebas
"""

import pandas as pd

# Paso 1: Leer el archivo Excel
df = pd.read_excel('dados_saude_conteo.xlsx')

# Paso 2: Filtrar los datos para el año 2009 y mes 1 y crear agrupaciones basadas en los 10 con más establecimientos
filtro_2009_1 = df[(df['anio'] == 2020) & (df['mes'] == 4)]
# Encontrar los locales con valor cero y eliminarlos
locales_con_valor = filtro_2009_1[filtro_2009_1['conteo'] != 0]
# Ordenar por 'locais' y crear grupos de 10
locales_con_valor = locales_con_valor.sort_values(by='conteo', ascending=False)
locales_con_valor['grupo'] = (locales_con_valor.reset_index().index // 10) + 1

# Mapear estos grupos a todos los registros correspondientes en el dataframe original
df = df.merge(locales_con_valor[['id', 'grupo']], on='id', how='left')
# Asumimos que los barrios que no estaban en 2009-1 se colocan en un grupo aparte
df['grupo'] = df['grupo'].fillna(max(locales_con_valor['grupo']) + 1)

# Paso 3: Agrupar por año, mes y grupo para obtener el total de establecimientos por grupo y por mes/año
df['Total'] = df.groupby(['anio', 'mes', 'grupo'])['conteo'].transform('sum')

# Paso 4: Calcular el porcentaje de participación de cada grupo en cada mes
df['Porcentaje_Participacion_Grupo'] = (df['Total'] / df.groupby(['anio', 'mes'])['conteo'].transform('sum')) * 100

# Paso 5: Calcular el HHI por cada mes/año usando los grupos
# Primero calculamos la cuota de mercado al cuadrado de cada grupo
df['Cuota_Cuadrada_Grupo'] = df['Porcentaje_Participacion_Grupo'] ** 2

# Agrupamos por mes y año para obtener el HHI por grupo
HHI_grupo = df.groupby(['anio', 'mes'])['Cuota_Cuadrada_Grupo'].sum().reset_index()

# Paso 6: Unir el HHI calculado con el dataframe original
df = df.merge(HHI_grupo, on=['anio', 'mes'], suffixes=('', '_HHI_Grupo'))

# Paso 7: Exportar los resultados a un nuevo archivo Excel
df.to_excel('resultado_con_hhi_grupo_ivs.xlsx', index=False)
HHI_grupo.to_excel('hhi_grupo_ivs.xlsx', index=False)

print("Archivo generado: 'resultado_con_hhi_grupo.xlsx'")
