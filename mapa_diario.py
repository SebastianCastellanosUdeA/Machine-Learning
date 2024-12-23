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

import imageio.v2 as imageio

# Carga los polígonos de los barrios
barrios = gpd.read_file("BairrosFortal.gpkg")

datos_pm25 = pd.read_excel("powevi.xlsx")
datos_pm25 = pd.read_excel("powerviprueba.xlsx")

datos_pm25['fecha'] = pd.to_datetime(datos_pm25['data'])

# Asegúrate de que los nombres de los barrios coincidan en ambos dataframes
barrios = barrios.merge(datos_pm25, how="left", left_on="nome", right_on="bairro")
x_min, y_min, x_max, y_max = barrios.total_bounds

fechas = datos_pm25['fecha'].unique()
cmap = LinearSegmentedColormap.from_list("custom_green_red", ["#FFB6C1", "#8B0000"], N=256)


for fecha in fechas:
    # Filtrar datos para un solo día
    datos_dia = barrios[barrios['fecha'] == fecha]
    
    # Crear el mapa
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    datos_dia.plot(column='pm25', ax=ax, legend=True, cmap='OrRd',
                   vmin=6, vmax=18,  # Fijar los valores mínimo y máximo de la escala de colores
                   legend_kwds={'label': "Concentração de PM2.5", 'orientation': "horizontal"})
    ax.set_title(f'PM2.5 {fecha.strftime("%Y-%m-%d")}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.savefig(f'mapa_{fecha.strftime("%Y-%m-%d")}.png')
    plt.close()
    

# Directorio donde están tus imágenes
directorio_imagenes = 'mapas'
# Obtiene una lista de archivos y los ordena
archivos_imagenes = sorted(glob.glob(os.path.join(directorio_imagenes, 'mapa_*.png')))

# Asume que todas las imágenes tienen el mismo tamaño
imagen_ejemplo = cv2.imread(archivos_imagenes[0])
altura, ancho, capas = imagen_ejemplo.shape


# Define el codec y crea el objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # También puedes usar 'XVID'
salida_video = cv2.VideoWriter('video_pm25.mp4', fourcc, 5, (ancho, altura))


# Lee cada imagen y la añade al video
for archivo in archivos_imagenes:
    imagen = cv2.imread(archivo)
    salida_video.write(imagen)

# Cierra el archivo de video
salida_video.release()
