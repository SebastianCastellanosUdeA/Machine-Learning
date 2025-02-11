# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 07:09:55 2024

@author: sebas
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import cv2
import os
import glob

# Carga los polígonos de los barrios
barrios = gpd.read_file("blz2.gpkg")

#datos = pd.read_excel("calculo_pm25_hexagono.xlsx")
datos = pd.read_excel("sin centro.xlsx")

#datos['fecha'] = pd.to_datetime(datos['data'])
datos['ano'] = pd.to_numeric(datos['year'])
datos['mes'] = pd.to_numeric(datos['month'])
# Asegúrate de que los nombres de los barrios coincidan en ambos dataframes
barrios = barrios.merge(datos, how="left", left_on="id", right_on="id")
x_min, y_min, x_max, y_max = barrios.total_bounds

#fechas = datos['fecha'].unique()
#fechas = datos['ano'].unique()
#combinaciones_unicas = datos[['fecha', 'periodo']].drop_duplicates()
combinaciones_unicas = datos[['mes', 'ano']].drop_duplicates()


#for fecha in fechas:
for _, row in combinaciones_unicas.iterrows():
    #fecha = row['fecha']
    #periodo = row['periodo']
    ano = row['ano']
    mes = row['mes']
    # Filtrar datos para un solo día
    #datos_dia = barrios[barrios['fecha'] == fecha]
    #datos_dia = barrios[barrios['ano'] == fecha]
    #datos_dia = barrios[(barrios['fecha'] == fecha) & (barrios['periodo'] == periodo)]
    datos_dia = barrios[(barrios['ano'] == ano) & (barrios['mes'] == mes)]

    # Crear el mapa
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    datos_dia.plot(column='conteo', ax=ax, legend=True, cmap='OrRd',
                   vmin=1, vmax=120,  # Fijar los valores mínimo y máximo de la escala de colores
    #ax.set_title(f'PM2.5 {fecha.strftime("%Y-%m-%d")}')
                   legend_kwds={'label': "atacados", 'orientation': "horizontal"})
    ax.set_title(f'atacado {ano} - {mes}')
    #ax.set_title(f'PM2.5 {fecha}')
    #ax.set_title(f'PM2.5 {fecha.strftime("%Y-%m-%d")} - {periodo}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    #plt.savefig(f'mapa_{fecha.strftime("%Y-%m-%d")}.png')
    #plt.savefig(f'mapa_{fecha}.png')
    plt.savefig(f'atacado_{ano}_{mes}.png')
    #plt.savefig(f'mapa_{fecha.strftime("%Y-%m-%d")}_{periodo}.png')
    plt.close()
    
    '''    
    # Crear el mapa compuesto
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 3 subplots en una fila

    # Mapa para PM2.5
    datos_dia.plot(column='pm25', ax=axs[0], legend=True, cmap='OrRd',
                    vmin=10, vmax=16, 
                    #vmin=datos_dia['pm25'].min(), vmax=datos_dia['pm25'].max(),
                    legend_kwds={'label': "Concentração de PM2.5", 'orientation': "horizontal"})
    axs[0].set_title(f'PM2.5 {fecha}')
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(y_min, y_max)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')

    # Mapa para IVS
    datos_dia.plot(column='ivs', ax=axs[1], legend=True, cmap='Blues',
                    vmin=0, vmax=0.9,
                    legend_kwds={'label': "Índice de Vulnerabilidade Social", 'orientation': "horizontal"})
    axs[1].set_title(f'IVS {fecha}')
    axs[1].set_xlim(x_min, x_max)
    axs[1].set_ylim(y_min, y_max)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')

    # Mapa para Locais
    datos_dia.plot(column='atacados', ax=axs[2], legend=True, cmap='Greens',
                    vmin=0, vmax=150,
                    legend_kwds={'label': "Atacados", 'orientation': "horizontal"})
    axs[2].set_title(f'atacados {fecha}')
    axs[2].set_xlim(x_min, x_max)
    axs[2].set_ylim(y_min, y_max)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].set_xlabel('')
    axs[2].set_ylabel('')
    plt.savefig(f'mapa_composto_{fecha}.png')
    plt.close()
    '''
# Directorio donde están tus imágenes
directorio_imagenes = 'mapas2'
# Obtiene una lista de archivos y los ordena
archivos_imagenes = sorted(glob.glob(os.path.join(directorio_imagenes, 'mapa_composto_*.png')))

# Asume que todas las imágenes tienen el mismo tamaño
imagen_ejemplo = cv2.imread(archivos_imagenes[0])
altura, ancho, capas = imagen_ejemplo.shape


# Define el codec y crea el objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # También puedes usar 'XVID'
salida_video = cv2.VideoWriter('video_consolidado_nao_maiores.mp4', fourcc, 1, (ancho, altura))


# Lee cada imagen y la añade al video
for archivo in archivos_imagenes:
    imagen = cv2.imread(archivo)
    salida_video.write(imagen)

# Cierra el archivo de video
salida_video.release()
