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


asset_path = 'projects/ee-projectsebas/assets/limites_fortaleza'
Contorno = ee.FeatureCollection(asset_path)

# Configurações iniciais para testar os 6 primeiros meses de 2024
startYear = 2020 # Ano inicial
endYear = 2020 # Ano final
endMonthInEndYear = 9 # Limitar até o mês de junho de 2024
region = Contorno # Área de interesse (Definir a geometria da sua região de interesse)
maxPixelsExport = 1e13 # Limite de pixels para exportação

# Função para gerar datas formatadas com dois dígitos
def formatDate(year, month, day):
  return str(year) + '-' + ('0' + str(month))[-2:] + '-' + ('0' + str(day))[-2:] 

# Função para calcular a média mensal do NO2
def getMonthlyAerosolImage(year, month):
    firstDate = ee.Date(f'{year}-{month:02d}-01') # Data inicial do mês
    stepDate = date(year, month, 1)
    # monthrange retorna o último dia do mês, basta setá-lo na data e pronto
    lastDate = ee.Date(str(stepDate.replace(day=monthrange(stepDate.year, stepDate.month)[1])))
  
    # Filtra a coleção por data e calcula a média mensal
    monthlyImage = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_AER_AI').select('absorbing_aerosol_index').filterDate(firstDate, lastDate).mean()
  
    return monthlyImage.clip(region); # Retorna a imagem recortada para a região

# Função para exportar imagens
def exportImage(image, year, month):
    exportName = 'Aerosol_Fortaleza_' + str(year) + '_' + ('0' + str(month))[-2:] 
    print(f'Começando a exportar {exportName}. . .')
    task = ee.batch.Export.image.toDrive(
        image = image, 
        description = exportName, 
        folder = "GEE_Aerosol",
        region = region.geometry(), 
        scale = 10,
        maxPixels = maxPixelsExport
        )
    task.start()

# Loop pelos anos e meses para exportar as imagens
year = startYear

while year <= endYear:
    if year != endYear: # Define o mês final
        lastMonth = 2
    else:
        lastMonth = 2
    month = 1
    while month <= lastMonth:
        monthlyImage = getMonthlyAerosolImage(year, month); # Obtém a imagem média mensal
        exportImage(monthlyImage, year, month); # Exporta a imagem
        month += 1
    year += 1


# In[ ]:




