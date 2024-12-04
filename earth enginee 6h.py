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
#ultinmo eepseba

# In[7]:


asset_path = 'projects/ee-projectsebas/assets/limites_fortaleza'
asset_path = 'projects/ee-sebastian-e2/assets/limites_fortaleza'
Contorno = ee.FeatureCollection(asset_path)

# Configurações iniciais para testar os 6 primeiros meses de 2024
startYear = 2014 # Ano inicial
endYear = 2014 # Ano final
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
    intervalImage = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2').select('tropospheric_NO2_column_number_density').filterDate(firstDate, lastDate).mean()
    return intervalImage.clip(region)  # Retorna la imagen recortada

def getHumidityImage(year, month, day, interval):
    firstDate, lastDate = formatDate(year, month, day, interval)
    
    # Filtrar la colección y calcular la media para el intervalo
    collection = ee.ImageCollection('NASA/GLDAS/V20/NOAH/G025/T3H') \
        .filterDate(firstDate, lastDate)
    
    image = collection.mean()
    
    # Calcular la humedad relativa usando la fórmula
    relativeHumidity = image.expression(
        '0.263 * p * q * (exp(17.67 * (T - T0) / (T - 29.65))) ** -1', {
            'T': image.select('Tair_f_inst'),  # Temperatura del aire
            'T0': 273.16,                     # Cero absoluto en Kelvin
            'p': image.select('Psurf_f_inst'), # Presión de superficie
            'q': image.select('Qair_f_inst')   # Humedad específica
        }
    ).float()
    
    return relativeHumidity.clip(region)

def exportImage(image, year, month, day, interval):
    exportName = f'humidity_{year}_{month:02d}_{day:02d}_{interval}'
    print(f'Comenzando a exportar {exportName}...')
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=exportName,
        folder="humidity",
        region=region.geometry(),
        crs='EPSG:4326',
        scale=100,  # Escala en metros (GLDAS tiene resolución baja)
        maxPixels=maxPixelsExport
    )
    task.start()




for year in range(startYear, endYear + 1):
    for month in range(12, 13):  # Todos los meses del año
        days_in_month = monthrange(year, month)[1]
        for day in range(1, days_in_month + 1):  # De 1 al último día del mes
            for interval in ['morning','night','midday','afternoon']:
                humidityImage = getHumidityImage(year, month, day, interval)
                exportImage(humidityImage, year, month, day, interval)
                #intervalImage = getIntervalAerosolImage(year, month, day, interval)  # Obtiene la imagen para el intervalo
                #exportImage(intervalImage, year, month, day, interval)  # Exporta la imagen