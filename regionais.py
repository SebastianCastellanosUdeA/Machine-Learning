# -*- coding: utf-8 -*-
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import box

# Cargar las capas de polígonos
barrios = gpd.read_file("blz.gpkg")  # Capa inicial con más de 150 polígonos
subdivisiones = gpd.read_file("regiones.gpkg")  # Nueva capa con 12 subdivisiones

# Cargar los datos y asociarlos a los barrios
datos = pd.read_excel("consolidado.xlsx")
datos['ano'] = pd.to_numeric(datos['año'])
datos['mes'] = pd.to_numeric(datos['mes'])

# Unir los datos con los polígonos de barrios
barrios = barrios.merge(datos, how="left", left_on="id", right_on="id")

# Obtener combinaciones únicas de año y mes
combinaciones_unicas = datos[['mes', 'ano']].drop_duplicates()

# Iterar sobre cada subdivisión
for i, subdivision in subdivisiones.iterrows():
    subdivision_geom = subdivision.geometry  # Geometría de la subdivisión

    # Recortar los polígonos de barrios dentro de la subdivisión
    barrios_recortados = barrios[barrios.intersects(subdivision_geom)].copy()
    barrios_recortados['geometry'] = barrios_recortados.intersection(subdivision_geom)

    # Obtener límites de la subdivisión para ajustar la vista
    x_min, y_min, x_max, y_max = subdivision_geom.bounds

    # Iterar sobre cada combinación de año y mes
    for _, row in combinaciones_unicas.iterrows():
        ano = row['ano']
        mes = row['mes']

        # Filtrar datos para el mes y año actual
        datos_dia = barrios_recortados[(barrios_recortados['ano'] == ano) & (barrios_recortados['mes'] == mes)]

        # Crear el mapa compuesto con 3 gráficos
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Mapa para PM2.5
        datos_dia.plot(column='PM', ax=axs[0], legend=True, cmap='OrRd', vmin=6, vmax=18,
                       legend_kwds={'label': "Concentração de PM2.5", 'orientation': "horizontal"})
        axs[0].set_title(f'PM2.5 {ano} - {mes}')
        axs[0].set_xlim(x_min, x_max)
        axs[0].set_ylim(y_min, y_max)
        axs[0].axis('off')

        # Mapa para IVS
        datos_dia.plot(column='IVS', ax=axs[1], legend=True, cmap='Blues', vmin=0, vmax=1,
                       legend_kwds={'label': "Índice de Vulnerabilidade Social", 'orientation': "horizontal"})
        axs[1].set_title(f'IVS {ano} - {mes}')
        axs[1].set_xlim(x_min, x_max)
        axs[1].set_ylim(y_min, y_max)
        axs[1].axis('off')

        # Mapa para Locais
        datos_dia.plot(column='locais', ax=axs[2], legend=True, cmap='Greens', vmin=0, vmax=120,
                       legend_kwds={'label': "Atacados", 'orientation': "horizontal"})
        axs[2].set_title(f'Atacados {ano} - {mes}')
        axs[2].set_xlim(x_min, x_max)
        axs[2].set_ylim(y_min, y_max)
        axs[2].axis('off')

        # Guardar el gráfico
        plt.savefig(f'mapa_composto_{ano}_{mes}_subdivision_{i}.png')
        plt.close()
