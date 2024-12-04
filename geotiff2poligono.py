# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:05:10 2024

@author: sebas
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask

barrios_gpkg = 'BairrosFortal.gpkg'
barrios_gdf = gpd.read_file(barrios_gpkg)

# Asegurarse de que las coordenadas del shapefile y el GeoTIFF estén en el mismo CRS
barrios_gdf = barrios_gdf.to_crs(epsg=4326)  # Convertir a EPSG:4326 si no está ya

# Listar todos los archivos .tif en la carpeta
tif_folder = 'tif2'
tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]

# Dividir los archivos en 4 partes
num_archivos = len(tif_files)
chunk_size = num_archivos // 2 # Tamaño de cada sublista (1/4 del total)

# Asegurar que no haya errores con los límites
tif_chunks = [
    tif_files[i:i + chunk_size] for i in range(0, num_archivos, chunk_size)
]

# Si el último grupo tiene menos de chunk_size, lo completamos
if len(tif_chunks) > 2:
    tif_chunks[-2].extend(tif_chunks[-1])  # Combinar el penúltimo con el último
    tif_chunks = tif_chunks[:-1]  # Eliminar el último grupo extra




# Iterar sobre las partes
for part_idx, chunk in enumerate(tif_chunks):
    print(f"\nProcesando la parte {part_idx + 1} de 2 (Archivos {len(chunk)} archivos)")
   
    # Guardar los resultados
    results = []
    
    # Iterar sobre cada archivo TIFF
    for tif_file in chunk:
        # Abrir el archivo GeoTIFF
        
        with rasterio.open(os.path.join(tif_folder, tif_file)) as geo_tiff:
            # Iterar sobre cada barrio (polígono) en el shapefile
            nodata_value = geo_tiff.nodata
            for idx, barrio in barrios_gdf.iterrows():
                
                # Extraer el polígono del barrio
                barrio_polygon = [barrio['geometry']]
    
                # Enmascarar el raster con el polígono del barrio
                out_image, out_transform = mask(geo_tiff, barrio_polygon, crop=True)
    
                # Convertir el array en un formato manejable (matriz 2D de valores de píxeles)
                out_image = out_image[0]  # Tomar la primera banda (suponiendo que es una imagen monocromática)
    
                # Si no hay un valor nodata explícito, tratamos el 0 como nodata
                if nodata_value is None:
                    # Reemplazar 0 por NaN (suponiendo que 0 es considerado vacío)
                    out_image[out_image == 0] = np.nan
                else:
                    # Si el valor nodata está definido, usamos ese valor
                    out_image[out_image == nodata_value] = np.nan
    
                # Extraer los valores no-nulos dentro del polígono (valores válidos)
                valores_validos = out_image[~np.isnan(out_image)]
                
                # Si hay valores válidos, calcular el promedio
                if len(valores_validos) > 0:
                    promedio = np.mean(valores_validos)
                else:
                    promedio = np.nan
    
                # Guardar el resultado
                results.append({
                    'Barrio': barrio['nome'],  # Asumiendo que hay una columna 'nombre' en el shapefile
                    'Archivo': tif_file,
                    'Promedio_Valor_Pixel': promedio
                })
    
    # Crear un DataFrame con los resultados
    results_df = pd.DataFrame(results)
    
    excel_filename = f'vviento2_{part_idx + 1}.xlsx'
    # Guardar los resultados en un archivo Excel
    results_df.to_excel(excel_filename, index=False)
