#Importações
import os
import pandas as pd
from geotiff import GeoTiff
import numpy as np


# Ler os pontos
pontos_df = pd.read_csv("pontos.csv")

# Listar todos os arquivos .tif na pasta
tif_folder = 'tif'
tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]

# salvar os resultados
results = []

# Iterar sobre cada arquivo TIFF
for tif_file in tif_files:
    #criar geotiff
    geo_tiff = GeoTiff(os.path.join(tif_folder, tif_file),crs_code=4326)
    
    #iterar por cada ponto
    for p in range(8):
        coord = pontos_df.iloc[p]
        array = []
        lon_array = []
        lat_array = []
        lat_vector = []
        lon_vector = []
        lat_vector_def = []
        lon_vector_def = []
        lat_coor = 0
        lon_coor = 0
        pixel = np.nan
        
        #criar array com os pixels
        zarr_array = geo_tiff.read()
        array = np.array(zarr_array)
        #criar array com latitude e longitude
        lon_array, lat_array = geo_tiff.get_coord_arrays()
        lat_vector = lat_array[:, 0]
        lon_vector = lon_array[0, :]
        
        #criar vetores com as latitudes e as longitudes
        lat_vector_def = np.append(lat_vector, geo_tiff.tif_bBox_converted[1][1])
        lon_vector_def = np.append(lon_vector, geo_tiff.tif_bBox_converted[1][0])
        lat_coor = coord['Latitude']
        lon_coor = coord['Longitude']
        
        #olhar as coordenadas em qual ponto encaixam 
        for i in range(len(lat_vector_def)):
            if lat_coor > lat_vector_def[i]:
                lat_i = i - 1
                break

        for j in range(len(lon_vector_def)):
            if lon_coor < lon_vector_def[j]:
                lon_j = j - 1
                break
        
        #obter o valor do pixel
        pixel = array[lat_i,lon_j]
        
        #salvar os resultados
        if not np.isnan(pixel):
            results.append({'File': tif_file,'Index': coord['I'],'Valor': pixel})
            
#criar excel
results_df = pd.DataFrame(results)
    
results_df.to_excel('resultados055.xlsx', index=False)
    













    
    
    
    
    
    



