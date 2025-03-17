# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 18:20:42 2025

@author: sebas
"""

import pandas as pd

# Paso 1: Leer el archivo Excel
# Asegúrate de cargar el archivo correcto (reemplaza 'archivo.xlsx' con el nombre de tu archivo)
df = pd.read_excel('dados_saude_conteo.xlsx')

# Paso 2: Agrupar por año y mes para obtener el total de establecimientos en cada mes/año
df['Total'] = df.groupby(['anio', 'mes'])['conteo'].transform('sum')

# Paso 3: Calcular el porcentaje de participación de cada barrio en cada mes
df['Porcentaje_Participacion'] = (df['conteo'] / df['Total']) * 100

# Paso 4: Calcular el HHI por cada mes/año
# Primero calculamos la cuota de mercado al cuadrado
df['Cuota_Cuadrada'] = ((df['conteo'] / df['Total'])*100) ** 2

# Agrupamos por mes y año para obtener el HHI
HHI = df.groupby(['anio', 'mes'])['Cuota_Cuadrada'].sum().reset_index()

# Paso 5: Unir el HHI calculado con el dataframe original
df = df.merge(HHI, on=['anio', 'mes'], suffixes=('', '_HHI'))

# Paso 6: Exportar los resultados a un nuevo archivo Excel
df.to_excel('resultado_con_hhi_ivs.xlsx', index=False)
HHI.to_excel('hhi_ivs.xlsx', index=False)

print("Archivo generado: 'resultado_con_hhi.xlsx'")
