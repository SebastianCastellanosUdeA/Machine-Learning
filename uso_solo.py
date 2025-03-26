# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 19:40:19 2025

@author: sebas
"""

import geopandas as gpd
import pandas as pd

# Leer los shapefiles
blz = gpd.read_file("blz.gpkg")  # Barrios
uso_solo = gpd.read_file("uso_solo.gpkg")  # Polígonos de uso de suelo

# Crear una lista de categorías de uso de suelo
categorias = uso_solo['uso_solo'].unique()

blz['geometry'] = blz['geometry'].apply(lambda x: x.buffer(0) if not x.is_valid else x)
uso_solo['geometry'] = uso_solo['geometry'].apply(lambda x: x.buffer(0) if not x.is_valid else x)




# Crear un DataFrame para almacenar los resultados
resultados = []

# Iterar sobre cada barrio (hexágono)
for _, barrio in blz.iterrows():
    id_barrio = barrio['id']  # O el nombre/ID del barrio
    
    # Obtener los polígonos de uso de suelo que se encuentran dentro del barrio
    intersecciones = uso_solo[uso_solo.geometry.intersects(barrio.geometry)]
    
    # Calcular el área total del barrio
    area_barrio = barrio.geometry.area
    
    # Inicializar un diccionario para almacenar los resultados de cada categoría
    conteo_categorias = {cat: 0 for cat in categorias}
    conteo_no_categoria = 0
    
    # Calcular el área de cada categoría dentro del barrio
    for _, poligono in intersecciones.iterrows():
        categoria = poligono['uso_solo']
        interseccion_area = barrio.geometry.intersection(poligono.geometry).area
        
        if interseccion_area > 0:
            # Si el área de la intersección es mayor a 0, se suma
            if categoria in conteo_categorias:
                conteo_categorias[categoria] += interseccion_area
            else:
                conteo_no_categoria += interseccion_area
    
    # Verificar que las áreas sumen correctamente y corregir si hay un problema
    area_total_categoria = sum(conteo_categorias.values()) + conteo_no_categoria
    if area_total_categoria > area_barrio:
        # Si la suma de las áreas supera el área del barrio, ajustar
        exceso = area_total_categoria - area_barrio
        for categoria in conteo_categorias:
            conteo_categorias[categoria] -= exceso * (conteo_categorias[categoria] / area_total_categoria)
        conteo_no_categoria -= exceso * (conteo_no_categoria / area_total_categoria)
    
    # Calcular el porcentaje de cada categoría en el barrio
    resultados_barrio = {
        'id_barrio': id_barrio,
        'residencial': conteo_categorias.get('Residencial', 0) / area_barrio * 100,
        'no_residencial': conteo_categorias.get('Não Residencial', 0) / area_barrio * 100,
        'Misto': conteo_categorias.get('Misto', 0) / area_barrio * 100,
        'Territorial': conteo_categorias.get('Territorial', 0) / area_barrio * 100
    }
    
    resultados.append(resultados_barrio)

# Crear un DataFrame con los resultados
df_resultados = pd.DataFrame(resultados)


# Exportar a Excel
df_resultados.to_excel("porcentaje_uso_suelo_por_barrio_corregido.xlsx", index=False)











# Crear un DataFrame para almacenar los resultados
resultados = []

# Iterar sobre cada barrio (hexágono)
for _, barrio in blz.iterrows():
    id_barrio = barrio['id']  # O el nombre/ID del barrio
    
    # Obtener los polígonos de uso de suelo que se encuentran dentro del barrio
    intersecciones = uso_solo[uso_solo.geometry.intersects(barrio.geometry)]
    
    # Calcular el área total del barrio
    area_barrio = barrio.geometry.area
    
    # Inicializar un diccionario para almacenar los resultados de cada categoría
    conteo_categorias = {cat: 0 for cat in categorias}
    conteo_no_categoria = 0
    
    # Calcular el área de cada categoría dentro del barrio
    for _, poligono in intersecciones.iterrows():
        categoria = poligono['uso_solo']
        interseccion_area = barrio.geometry.intersection(poligono.geometry).area
        if categoria in conteo_categorias:
            conteo_categorias[categoria] += interseccion_area
        else:
            conteo_no_categoria += interseccion_area
    
    # Calcular el porcentaje de cada categoría en el barrio
    resultados_barrio = {
        'id_barrio': id_barrio,
        'residencial': conteo_categorias.get('Residencial', 0) / area_barrio * 100,
        'no_residencial': conteo_categorias.get('Não Residencial', 0) / area_barrio * 100,
        'categoria3': conteo_categorias.get('Misto', 0) / area_barrio * 100,
        'categoria4': conteo_categorias.get('Territorial', 0) / area_barrio * 100
    }
    
    resultados.append(resultados_barrio)

# Crear un DataFrame con los resultados
df_resultados = pd.DataFrame(resultados)

# Exportar a Excel
df_resultados.to_excel("porcentaje_uso_suelo_por_barrio.xlsx", index=False)
