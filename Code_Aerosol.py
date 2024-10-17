#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ee
import time
from datetime import date
from calendar import monthrange
from IPython.display import display


# In[5]:


ee.Authenticate()

ee.Initialize(project='ee-projectsebas')


# In[7]:


asset_path = 'projects/ee-projectsebas/assets/Contorno'
Contorno = ee.FeatureCollection(asset_path)

# Configurações iniciais para testar os 6 primeiros meses de 2024
startYear = 2022 # Ano inicial
endYear = 2022 # Ano final
endMonthInEndYear = 6 # Limitar até o mês de junho de 2024
region = ee.Geometry.Rectangle([-38.65, -3.9, -38.38, -3.68]) 
maxPixelsExport = 1e13 # Limite de pixels para exportação

# Función para generar fechas con formato correcto (añadida la hora)
def formatDate(year, month, day, hour):
    return f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00'

# Función para calcular la media horaria del NO2
def getHourlyAerosolImage(year, month, day, hour):
    # Fecha inicial y final por hora
    firstDate = ee.Date(formatDate(year, month, day, hour)) # Hora específica
    lastDate = firstDate.advance(1, 'hour')  # Avanzar una hora
    
    
    # Filtrar la colección por la hora y calcular la media horaria
    hourlyImage = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2') \
                    .select('tropospheric_NO2_column_number_density') \
                    .filterDate(firstDate, lastDate) \
                    .mean()

    return hourlyImage.clip(region)  # Retorna la imagen recortada para la región


# Función para exportar imágenes
def exportImage(image, year, month, day, hour):
    exportName = f'NO2_{year}_{month:02d}_{day:02d}_{hour:02d}h'
    print(f'Comenzando a exportar {exportName}...')
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=exportName,
        folder="GEE_no2",
        region=region,
        crs='EPSG:4326',
        maxPixels=maxPixelsExport
    )
    task.start()

# Loop por años, meses, días y horas para exportar imágenes horarias
for year in range(startYear, endYear + 1):
    for month in range(1, 2):# De enero a diciembre
        # Determina los días del mes
        days_in_month = monthrange(year, month)[1]
        #for day in range(1, days_in_month + 1):  # De 1 al último día del mes
        for day in range(1, 2):
            for hour in range(0, 24):  # De 0 a 23 horas
                hourlyImage = getHourlyAerosolImage(year, month, day, hour)  # Obtiene la imagen horaria
                exportImage(hourlyImage, year, month, day, hour)  # Exporta la imagen




