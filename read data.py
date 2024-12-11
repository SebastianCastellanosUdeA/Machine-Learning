# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 19:03:14 2024

@author: sebas
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Cargar datos
dir_viento_pt1 = pd.read_excel("temperatura pt1.xlsx")
dir_viento_pt2 = pd.read_excel("um2.xlsx")

pt1_filtered = dir_viento_pt1[dir_viento_pt1["PERIODO"].isin(["night"])]

# Filtrar los datos relevantes en pt2
pt2_filtered = dir_viento_pt2[dir_viento_pt2["PERIODO"].isin([ "midday", "afternoon"])]

# Pivotar para obtener night, midday, afternoon como columnas
pt2_pivot = pt2_filtered.pivot_table(index=["BAIRRO", "ANO", "MES", "DIA"],
                                     columns="PERIODO",
                                     values="VALOR").reset_index()

# Eliminar filas con valores nulos en night, midday o afternoon
pt2_pivot = pt2_pivot.dropna(subset=["night", "midday", "afternoon"])

# Crear un diccionario para guardar los modelos y resultados por barrio
model_results = {}

# Crear un gráfico por cada barrio
for bairro, group in pt2_pivot.groupby("BAIRRO"):
    # Calcular la diferencia entre midday y night
    group["difference"] = group["afternoon"] - group["night"]
    
    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    plt.scatter(group["night"], group["difference"], label=f"BAIRRO: {bairro}")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title(f"Diferencia entre Night y Midday para {bairro}")
    plt.xlabel("Night")
    plt.ylabel("Diferencia (Midday - Night)")
    plt.legend()
    plt.grid(True)
    plt.show()
    


# Calcular la diferencia
pt2_pivot["difference"] = pt2_pivot["afternoon"] - pt2_pivot["night"]

# Calcular el promedio de la diferencia por barrio
average_difference_by_bairro = pt2_pivot.groupby("BAIRRO")["difference"].mean().reset_index()

# Renombrar columnas para claridad
average_difference_by_bairro.rename(columns={"difference": "average_difference"}, inplace=True)

# Mostrar los resultados
print(average_difference_by_bairro)

# Opcional: Guardar los resultados en un archivo Excel
average_difference_by_bairro.to_excel("average_difference_by_bairro.xlsx", index=False)
print("Promedio de diferencias por barrio guardado en 'average_difference_by_bairro.xlsx'")

# Crear un gráfico por cada barrio
for bairro, group in pt2_pivot.groupby("BAIRRO"):
    # Verificar nombres de columnas
    print(f"Columnas en el grupo {bairro}: {group.columns}")
    
    # Calcular la diferencia entre midday y night
    group["difference"] = group["afternoon"] - group["night"]
    
    # Asegurarse de que las columnas sean numéricas
    group["ANO"] = group["ANO"].astype(int)
    group["MES"] = group["MES"].astype(int)
    group["DIA"] = group["DIA"].astype(int)
    
    # Crear una columna de tiempo como combinación de año, mes y día
    try:
        group["time"] = pd.to_datetime(group[["ANO", "MES", "DIA"]].rename(columns={
            "ANO": "year", "MES": "month", "DIA": "day"
        }))
    except Exception as e:
        print(f"Error al crear la columna de tiempo para el barrio {bairro}: {e}")
        continue
    
    # Crear el gráfico
    plt.figure(figsize=(12, 6))
    plt.scatter(group["time"], group["night"], color='blue', label="Night (VALOR)", alpha=0.7)
    plt.scatter(group["time"], group["midday"], color='red', label="midday", alpha=0.7)
    plt.scatter(group["time"], group["difference"], color='green', label="Diferencia (M)", alpha=0.7)
    
    # Configurar títulos y etiquetas
    plt.title(f"Comportamiento Temporal para {bairro}")
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()

    
# Agrupar los datos por "BAIRRO"
for bairro, group in pt2_pivot.groupby("BAIRRO"):
    print(f"\nProcesando modelo para BAIRRO: {bairro}")
    
    # Extraer las variables para este barrio
    X_night = group[["night"]]
    y_midday = group["midday"]
    y_afternoon = group["afternoon"]
    
    # Crear y ajustar modelos de regresión
    model_midday = LinearRegression()
    model_afternoon = LinearRegression()
    
    model_midday.fit(X_night, y_midday)
    model_afternoon.fit(X_night, y_afternoon)
    
    # Calcular el R^2 para cada modelo
    r2_midday = model_midday.score(X_night, y_midday)
    r2_afternoon = model_afternoon.score(X_night, y_afternoon)
    
    # Guardar los resultados
    model_results[bairro] = {
        "midday": {
            "model": model_midday,
            "coef": model_midday.coef_[0],
            "intercept": model_midday.intercept_,
            "r2": r2_midday,
        },
        "afternoon": {
            "model": model_afternoon,
            "coef": model_afternoon.coef_[0],
            "intercept": model_afternoon.intercept_,
            "r2": r2_afternoon,
        },
    }
    
    # Imprimir los resultados para este barrio
    print(f"Modelo Midday: Midday = {model_midday.coef_[0]:.3f} * Night + {model_midday.intercept_:.3f}, R^2 = {r2_midday:.3f}")
    print(f"Modelo Afternoon: Afternoon = {model_afternoon.coef_[0]:.3f} * Night + {model_afternoon.intercept_:.3f}, R^2 = {r2_afternoon:.3f}")

# Usar los modelos para predecir midday y afternoon para dir_viento_pt1
predicted_rows = []

# Iterar por "BAIRRO" en dir_viento_pt1
for bairro, group in dir_viento_pt1.groupby("BAIRRO"):
    if bairro in model_results:
        # Obtener el modelo para este barrio
        model_midday = model_results[bairro]["midday"]["model"]
        model_afternoon = model_results[bairro]["afternoon"]["model"]
        
        # Predecir midday y afternoon
        group = group.copy()
        group["midday"] = model_midday.predict(group[["VALOR"]])
        group["afternoon"] = model_afternoon.predict(group[["VALOR"]])
        
        # Agregar al resultado final
        predicted_rows.append(group)

# Combinar todos los grupos en un solo DataFrame
predicted_data = pd.concat(predicted_rows, ignore_index=True)

# Reestructurar los datos para guardar
final_data = pd.concat([
    predicted_data.rename(columns={"VALOR": "night"})[["BAIRRO", "ANO", "MES", "DIA", "PERIODO", "night"]],
    predicted_data.rename(columns={"midday": "VALOR", "PERIODO": "midday"})[["BAIRRO", "ANO", "MES", "DIA", "PERIODO", "VALOR"]],
    predicted_data.rename(columns={"afternoon": "VALOR", "PERIODO": "afternoon"})[["BAIRRO", "ANO", "MES", "DIA", "PERIODO", "VALOR"]],
])

# Guardar en un archivo Excel
final_data.to_excel("dir_viento_extrapolated_by_bairro.xlsx", index=False)

print("Datos extrapolados por barrio guardados en 'dir_viento_extrapolated_by_bairro.xlsx'")



pt2_pivot["difference_midday"] = pt2_pivot["midday"] - pt2_pivot["night"]
pt2_pivot["difference_afternoon"] = pt2_pivot["afternoon"] - pt2_pivot["night"]

# Calcular los promedios de las diferencias por barrio
average_differences = pt2_pivot.groupby("BAIRRO")[["difference_midday", "difference_afternoon"]].mean().reset_index()

# Unir los promedios con los datos de 2000-2020 (solo night)
pt1_filtered = pd.merge(pt1_filtered, average_differences, on="BAIRRO", how="left")

# Aplicar las diferencias promedio para calcular midday y afternoon
pt1_filtered["midday"] = pt1_filtered["VALOR"] + pt1_filtered["difference_midday"]
pt1_filtered["afternoon"] = pt1_filtered["VALOR"] + pt1_filtered["difference_afternoon"]

# Guardar el resultado en un archivo Excel
pt1_filtered.to_excel("dir_viento_extrapolated_2000_2020.xlsx", index=False)
print("Datos extrapolados guardados en 'dir_viento_extrapolated_2000_2020.xlsx'")





# Calcular las diferencias en los datos conocidos (2021-2023)
pt2_pivot["difference_midday"] = pt2_pivot["midday"] - pt2_pivot["night"]
pt2_pivot["difference_afternoon"] = pt2_pivot["afternoon"] - pt2_pivot["night"]

# Calcular los promedios de las diferencias por barrio
average_differences = pt2_pivot.groupby("BAIRRO")[["difference_midday", "difference_afternoon"]].mean().reset_index()

# Identificar el barrio con el menor promedio de dirección del viento
pt2_pivot["average_direction"] = pt2_pivot["night"].mean()
min_dir_bairro = pt2_pivot.groupby("BAIRRO")["average_direction"].mean().idxmin()
min_diff_midday = average_differences.loc[average_differences["BAIRRO"] == min_dir_bairro, "difference_midday"].values[0]
min_diff_afternoon = average_differences.loc[average_differences["BAIRRO"] == min_dir_bairro, "difference_afternoon"].values[0]

# Identificar los barrios en pt1 que no están en pt2
barrios_in_pt1 = pt1_filtered["BAIRRO"].unique()
barrios_in_pt2 = pt2_pivot["BAIRRO"].unique()
missing_barrios = [b for b in barrios_in_pt1 if b not in barrios_in_pt2]

# Asignar los promedios del barrio con menor dirección del viento a los barrios faltantes
default_differences = pd.DataFrame({
    "BAIRRO": missing_barrios,
    "difference_midday": min_diff_midday,
    "difference_afternoon": min_diff_afternoon
})

# Unir los promedios con los datos de 2000-2020 (solo night)
average_differences = pd.concat([average_differences, default_differences], ignore_index=True)
pt1_filtered = pd.merge(pt1_filtered, average_differences, on="BAIRRO", how="left")

# Aplicar las diferencias promedio para calcular midday y afternoon
pt1_filtered["midday"] = pt1_filtered["VALOR"] + pt1_filtered["difference_midday"]
pt1_filtered["afternoon"] = pt1_filtered["VALOR"] + pt1_filtered["difference_afternoon"]

# Guardar el resultado en un archivo Excel
pt2_pivot.to_excel("umed1.xlsx", index=False)
print("Datos extrapolados guardados en 'dir_viento_extrapolated_2000_2020.xlsx'")



# Supongamos que tienes tus dataframes ya cargados en temp1 y temp2
# Si necesitas cargarlos desde un archivo Excel, usarías:
temp1 = pd.read_excel('umed2.xlsx')
temp2 = pd.read_excel('umed1.xlsx')

# Identificar los barrios presentes en temp1 y no en temp2
barrios_temp1 = set(temp1['BAIRRO'].unique())
barrios_temp2 = set(temp2['BAIRRO'].unique())


# Barrios presentes en ambos temp1 y temp2
barrios_comunes = barrios_temp1.intersection(barrios_temp2)
barrios_faltantes = barrios_temp1 - barrios_temp2

similares = pd.DataFrame(columns=['Barrio_faltante', 'Barrio_similar', 'Diferencia_promedio'])

for barrio in barrios_faltantes:
    diferencias = []
    datos_barrio_faltante = temp1[temp1['BAIRRO'] == barrio]

    for b in barrios_comunes:
        datos_barrio_comun = temp1[temp1['BAIRRO'] == b]
        comparacion = datos_barrio_faltante.merge(datos_barrio_comun, on=['ANO', 'MES', 'DIA'], suffixes=('_faltante', '_comun'))
        
        comparacion['diff_afternoon'] = abs(comparacion['afternoon_faltante'] - comparacion['afternoon_comun'])
        comparacion['diff_midday'] = abs(comparacion['midday_faltante'] - comparacion['midday_comun'])
        
        diff_promedio = comparacion[['diff_afternoon', 'diff_midday']].mean().mean()
        diferencias.append((b, diff_promedio))

    # Encontrar el barrio con menor diferencia promedio
    barrio_similar, min_diff = min(diferencias, key=lambda x: x[1])
    # Crear un DataFrame para la fila actual y concatenarlo con el DataFrame principal
    nueva_fila = pd.DataFrame({'Barrio_faltante': [barrio], 'Barrio_similar': [barrio_similar], 'Diferencia_promedio': [min_diff]})
    similares = pd.concat([similares, nueva_fila], ignore_index=True)
    

for _, row in similares.iterrows():
    barrio_faltante = row['Barrio_faltante']
    barrio_similar = row['Barrio_similar']
    datos_similares = temp2[temp2['BAIRRO'] == barrio_similar].copy()
    datos_similares['BAIRRO'] = barrio_faltante  # Reemplazar el barrio por el faltante
    temp2 = pd.concat([temp2, datos_similares], ignore_index=True)


temp2.to_excel("umed_extrapol.xlsx", index=False)
