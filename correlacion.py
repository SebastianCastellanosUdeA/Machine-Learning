# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 23:08:12 2025

@author: sebas
"""

import pandas as pd

# Cargar los datos desde un archivo Excel
df = pd.read_excel('sem centro ext.xlsx')

# Asegurarse de que las columnas 'año' y 'mes' son de tipo entero si no lo son
df['año'] = df['año'].astype(int)
df['mes'] = df['mes'].astype(int)

# Definir una función para aplanar la matriz de correlación
def flatten_corr(corr, year, month):
    corr = corr.reset_index()
    corr = pd.melt(corr, id_vars=['index'], value_vars=corr.columns[corr.columns != 'index'])
    corr.columns = ['variable1', 'variable2', 'valor_correlacion']
    corr['año'] = year
    corr['mes'] = month
    return corr

# Agrupar por 'año' y 'mes', calcular la correlación y aplanar los resultados
resultados = []
for (year, month), group in df.groupby(['año', 'mes']):
    corr_matrix = group[['IVS', 'locais', 'PM']].corr()
    flat_corr = flatten_corr(corr_matrix, year, month)
    resultados.append(flat_corr)

# Concatenar todos los DataFrames aplanados en uno solo
resultados_df = pd.concat(resultados)

# Filtrar para remover las correlaciones de una variable consigo misma (si es necesario)
resultados_df = resultados_df[resultados_df['variable1'] != resultados_df['variable2']]

# Imprimir los resultados finales
print(resultados_df)

# Opcional: guardar en Excel
resultados_df.to_excel('correlacion_sem_centro_ext.xlsx', index=False)
