# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 22:52:29 2025

@author: sebas
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime

# Carga el GeoPackage
poligonos = gpd.read_file("blz2.gpkg")
puntos = gpd.read_file("base_principal.gpkg")

print("Tipo de geometría en polígonos:", poligonos.geometry.type.unique())
print("Tipo de geometría en puntos:", puntos.geometry.type.unique())
print("Geometrías válidas en polígonos:", poligonos.geometry.is_valid.all())
print("Geometrías válidas en puntos:", puntos.geometry.is_valid.all())
print("CRS de polígonos:", poligonos.crs)
print("CRS de puntos:", puntos.crs)

codigos_cnae = ['3513100', '4511103', '4511104', '4511105', '4511106', 
                '4530701', '4530702', '4622200', '4541201', '4541202', 
                '4621400', '4623101', '4623102', '4623106', '4623108', 
                '4623109', '4623199', '4631100', '4632001', '4632002',
                '4632003', '4633801', '4633802', '4634601', '4634602', 
                '4634603', '4634699', '4635401', '4635402', '4635403', 
                '4635499', '4636201', '4636202', '4637101', '4637102', 
                '4637103', '4637104', '4637105', '4637106', '4637107',
                '4637199', '4639701', '4639702', '4641901', '4641902',
                '4641903', '4642701', '4642702', '4643501', '4643502',
                '4644301', '4644302', '4645101', '4645102', '4645103',
                '4646001', '4646002', '4647801', '4647802', '4649401', 
                '4649402', '4649403', '4649404', '4649405', '4649406', 
                '4649407', '4649408', '4649409', '4649410', '4649499', 
                '4651601', '4651602', '4652400', '4661300', '4662100', 
                '4663000', '4664800', '4665600', '4669901', '4669999', 
                '4671100', '4672900', '4673700', '4674500', '4679601', 
                '4679602', '4679603', '4679604', '4679699', '4681801', 
                '4681802', '4681804', '4681805', '4682600', '4683400', 
                '4684201', '4684299', '4685100', '4686901', '4686902', 
                '4687701', '4687702', '4687703', '4689301', '4689302', 
                '4689399', '4691500', '4692300', '4693100', '4633803', 
                '4681803', '4684202','4623103', '4623104', '4623105', '4623107']

resultados = pd.DataFrame()


# Iterar por cada año y cada mes
for year in range(2003, 2004):
    for month in range(1, 6):
        first_day = datetime(year, month, 1).strftime('%Y%m%d')
        last_day = datetime(year, month + 1, 1).strftime('%Y%m%d') if month < 12 else datetime(year + 1, 1, 1).strftime('%Y%m%d')
        
        # Filtrar puntos según los criterios de fecha y códigos CNAE
        puntos_filtrados = puntos[
            (puntos['cnae_principal_cod'].isin(codigos_cnae)) &
            (puntos['dt_sit_cadastral'] > first_day) &
            (puntos['dt_abertura_estab'] < last_day)
        ]

        puntos_dentro = gpd.sjoin(puntos_filtrados, poligonos, how="left", op='within')
        conteo_puntos = puntos_dentro.groupby('id').size()
        
        # Agregar los conteos a un DataFrame temporal con información del año y mes
        conteos_temp = poligonos[['geometry']].copy()
        conteos_temp['conteo_puntos'] = poligonos.index.map(conteo_puntos).fillna(0)
        conteos_temp['año'] = year
        conteos_temp['mes'] = month

        # Concatenar los resultados al DataFrame final
        resultados = pd.concat([resultados, conteos_temp], ignore_index=True)

# Guardar o mostrar los resultados
print(resultados.head())
resultados.to_excel('resu.xlsx', index=False)
