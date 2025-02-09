# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:49:09 2025

@author: sebas
"""

import pandas as pd
df1 = pd.read_excel("resultado_1.xlsx")
df2 = pd.read_excel("resultado_2.xlsx")
df3 = pd.read_excel("resultado_3.xlsx")
df4 = pd.read_excel("resultado_4.xlsx")
df5 = pd.read_excel("resultado_5.xlsx")
df6 = pd.read_excel("resultado_6.xlsx")
df7 = pd.read_excel("resultado_7.xlsx")
df8 = pd.read_excel("resultado_8.xlsx")
df9 = pd.read_excel("resultado_9.xlsx")
df10 = pd.read_excel("resultado_10.xlsx")




df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)


df['DATA_ATENDIMENTO'] = pd.to_datetime(df['DATA_ATENDIMENTO'], format='%d/%m/%Y %H:%M:%S')

# Extrae el año y el mes de la columna 'data_atendimento'
df['anio'] = df['DATA_ATENDIMENTO'].dt.year
df['mes'] = df['DATA_ATENDIMENTO'].dt.month

# Agrupa los datos por barrio, año y mes y cuenta las entradas
resultado = df.groupby(['id', 'anio', 'mes']).size().reset_index(name='conteo')
resultado.to_excel('dados_saude_conteo.xlsx', index=False)
