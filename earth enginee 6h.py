#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ee
import time
from datetime import date
from calendar import monthrange
from IPython.display import display
import os

# In[5]:

cred_path = os.path.expanduser("~/.config/earthengine/credentials")
if os.path.exists(cred_path):
    os.remove(cred_path)

ee.Authenticate()

ee.Initialize(project='ee-projectsebas')
ee.Initialize(project='ee-sebastian-e2')
#ultinmo ee2

# In[7]:


asset_path = 'projects/ee-projectsebas/assets/limites_fortaleza'
asset_path = 'projects/ee-sebastian-e2/assets/limites_fortaleza'
Contorno = ee.FeatureCollection(asset_path)

# Configurações iniciais para testar os 6 primeiros meses de 2024
startYear = 2021 # Ano inicial
endYear = 2021 # Ano final
#2023  también
maxPixelsExport = 1e13 # Limite de pixels para exportação
region = Contorno

# Función para generar fechas con formato correcto (añadida la hora)
def formatDate(year, month, day, interval):
    intervals = {
        'morning': ('07:00:00', 4),   # 7 am a 11 am
        'midday': ('11:00:00', 4),    # 11 am a 3 pm
        'afternoon': ('15:00:00', 4), # 3 pm a 7 pm
        'night': ('19:00:00', 12)     # 7 pm a 7 am del siguiente día
    }
    start_time, duration = intervals[interval]
    firstDate = ee.Date(f'{year:04d}-{month:02d}-{day:02d}T{start_time}')
    lastDate = firstDate.advance(duration, 'hour')
    return firstDate, lastDate

def getIntervalAerosolImage(year, month, day, interval):
    firstDate, lastDate = formatDate(year, month, day, interval)
    
    # Filtrar la colección por el intervalo y calcular la media
    intervalImage = ee.ImageCollection('NOAA/CFSV2/FOR6H').select('u-component_of_wind_height_above_ground').filterDate(firstDate, lastDate).mean()
    #Wind_f_inst vel viento 2021 a 2023
    #u viento 2021 a 2023: NOAA/CFSV2/FOR6H, u-component_of_wind_height_above_ground y mesmo v
    #uvento: u_component_of_wind_10m, colecction: ECMWF/ERA5/DAILY
    return intervalImage.clip(region)  # Retorna la imagen recortada


def exportImage(image, year, month, day, interval):
    exportName = f'wvt_{year}_{month:02d}_{day:02d}_{interval}'
    print(f'Comenzando a exportar {exportName}...')
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=exportName,
        folder="UViento",
        region=region.geometry(),
        crs='EPSG:4326',
        scale=10,
        maxPixels=maxPixelsExport
    )
    task.start()


for year in range(startYear, endYear + 1):
    for month in range(1, 13):  # Todos los meses del año
        days_in_month = monthrange(year, month)[1]
        for day in range(1, days_in_month + 1):  # De 1 al último día del mes
            #for interval in ['midday', 'night']:
            for interval in ['morning', 'midday', 'afternoon', 'night']:  # Intervalos
                intervalImage = getIntervalAerosolImage(year, month, day, interval)  # Obtiene la imagen para el intervalo
                exportImage(intervalImage, year, month, day, interval)  # Exporta la imagen