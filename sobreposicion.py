# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:51:57 2025

@author: sebas
"""

import geopandas as gpd
import pandas as pd

# Carga tus capas
regiones = gpd.read_file('regiones.gpkg')
hexagonos = gpd.read_file('blz.gpkg')

# Asegúrate de que ambas capas tienen el mismo CRS
regiones = regiones.to_crs(hexagonos.crs)
hexagonos['area_hexagono'] = hexagonos.area
# Intersección de las capas
interseccion = gpd.overlay(hexagonos, regiones, how='intersection')

# Calcula el área de la intersección y el área original de los hexágonos
interseccion['area_interseccion'] = interseccion.area
interseccion['porcentaje'] = (interseccion['area_interseccion'] / interseccion['area_hexagono']) * 100
#print(interseccion[['geometry', 'area_regione', 'area_interseccion', 'porcentaje']])


# Filtra las intersecciones donde el porcentaje es mayor a 25%
interseccion_filtrada = interseccion[interseccion['porcentaje'] > 25]


# Obtén la columna que identifica cada región, por ejemplo 'nombre'
nombre_region = 'regiao_adm'  # Asegúrate de cambiar esto por el nombre real de la columna

# Itera sobre cada región única en la intersección filtrada
for region in interseccion_filtrada[nombre_region].unique():
    # Filtra los hexágonos para esta región específica
    hexagonos_region = interseccion_filtrada[interseccion_filtrada[nombre_region] == region]
    
    # Exporta los hexágonos filtrados a un archivo GPKG
    nombre_archivo = f"hexagonos_{region}.gpkg"
    hexagonos_region.to_file(nombre_archivo, driver='GPKG')

    print(f'Archivo creado: {nombre_archivo}')

interseccion.to_excel('hrrr.xlsx', index=False)
