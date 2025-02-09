# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 22:43:59 2025

@author: sebas
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Carga el GeoPackage
poligonos = gpd.read_file("blz.gpkg")

# Carga el Excel
df = pd.read_excel("cadi.xlsx")

#21-808139
# Asegúrate de que las columnas de latitud y longitud están correctamente nombradas
df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)

puntos = gpd.GeoDataFrame(df, geometry='geometry')

# Asegúrate de que ambos GeoDataFrames usan el mismo sistema de referencia de coordenadas
puntos.set_crs(poligonos.crs, inplace=True)

# Realiza una operación de intersección espacial
puntos = gpd.sjoin(puntos, poligonos, how="left", op='within')

# Exporta el resultado a Excel
puntos.drop(columns='geometry').to_excel("resultado_cadi.xlsx", index=False)



df2 = pd.read_excel("grupo_2.xlsx")
df3 = pd.read_excel("grupo_3.xlsx")
df4 = pd.read_excel("grupo_4.xlsx")
df5 = pd.read_excel("grupo_5.xlsx")
df6 = pd.read_excel("grupo_6.xlsx")
df7 = pd.read_excel("grupo_7.xlsx")
df8 = pd.read_excel("grupo_8.xlsx")
df9 = pd.read_excel("grupo_9.xlsx")
df10 = pd.read_excel("grupo_10.xlsx")


df2['geometry'] = df2.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
df3['geometry'] = df3.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
df4['geometry'] = df4.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
df5['geometry'] = df5.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
df6['geometry'] = df6.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
df7['geometry'] = df7.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
df8['geometry'] = df8.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
df9['geometry'] = df9.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
df10['geometry'] = df10.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)


puntos2 = gpd.GeoDataFrame(df2, geometry='geometry')
puntos3 = gpd.GeoDataFrame(df3, geometry='geometry')
puntos4 = gpd.GeoDataFrame(df4, geometry='geometry')
puntos5 = gpd.GeoDataFrame(df5, geometry='geometry')
puntos6 = gpd.GeoDataFrame(df6, geometry='geometry')
puntos7 = gpd.GeoDataFrame(df7, geometry='geometry')
puntos8 = gpd.GeoDataFrame(df8, geometry='geometry')
puntos9 = gpd.GeoDataFrame(df9, geometry='geometry')
puntos10 = gpd.GeoDataFrame(df10, geometry='geometry')


puntos2.set_crs(poligonos.crs, inplace=True)
puntos3.set_crs(poligonos.crs, inplace=True)
puntos4.set_crs(poligonos.crs, inplace=True)
puntos5.set_crs(poligonos.crs, inplace=True)
puntos6.set_crs(poligonos.crs, inplace=True)
puntos7.set_crs(poligonos.crs, inplace=True)
puntos8.set_crs(poligonos.crs, inplace=True)
puntos9.set_crs(poligonos.crs, inplace=True)
puntos10.set_crs(poligonos.crs, inplace=True)

puntos2 = gpd.sjoin(puntos2, poligonos, how="left", op='within')
puntos3 = gpd.sjoin(puntos3, poligonos, how="left", op='within')
puntos4 = gpd.sjoin(puntos4, poligonos, how="left", op='within')
puntos5 = gpd.sjoin(puntos5, poligonos, how="left", op='within')
puntos6 = gpd.sjoin(puntos6, poligonos, how="left", op='within')
puntos7 = gpd.sjoin(puntos7, poligonos, how="left", op='within')
puntos8 = gpd.sjoin(puntos8, poligonos, how="left", op='within')
puntos9 = gpd.sjoin(puntos9, poligonos, how="left", op='within')
puntos10 = gpd.sjoin(puntos10, poligonos, how="left", op='within')


puntos2.drop(columns='geometry').to_excel("resultado_2.xlsx", index=False)
puntos3.drop(columns='geometry').to_excel("resultado_3.xlsx", index=False)
puntos4.drop(columns='geometry').to_excel("resultado_4.xlsx", index=False)
puntos5.drop(columns='geometry').to_excel("resultado_5.xlsx", index=False)
puntos6.drop(columns='geometry').to_excel("resultado_6.xlsx", index=False)
puntos7.drop(columns='geometry').to_excel("resultado_7.xlsx", index=False)
puntos8.drop(columns='geometry').to_excel("resultado_8.xlsx", index=False)
puntos9.drop(columns='geometry').to_excel("resultado_9.xlsx", index=False)
puntos10.drop(columns='geometry').to_excel("resultado_10.xlsx", index=False)
