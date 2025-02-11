# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:51:57 2025

@author: sebas
"""

import geopandas as gpd
import pandas as pd

# Carga tus capas
hexagonos = gpd.read_file('blz.gpkg')
barrios = gpd.read_file('fortal4326.gpkg')

# Asegúrate de que ambas capas tienen el mismo CRS
hexagonos = hexagonos.to_crs(barrios.crs)
barrios['area_barrio'] = barrios.area
# Intersección de las capas
interseccion = gpd.overlay(barrios, hexagonos, how='intersection')

# Calcula el área de la intersección y el área original de los hexágonos
interseccion['area_interseccion'] = interseccion.area
interseccion['porcentaje'] = (interseccion['area_interseccion'] / interseccion['area_barrio']) * 100
#print(interseccion[['geometry', 'area_hexagono', 'area_interseccion', 'porcentaje']])


interseccion.to_excel('hexagonosybarrios.xlsx', index=False)
