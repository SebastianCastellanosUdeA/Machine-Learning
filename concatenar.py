import pandas as pd

# Cargar los datos desde los archivos Excel
df_aerosol = pd.read_excel('aerosol.xlsx')
df_dirviento = pd.read_excel('dirviento.xlsx')
df_velviento = pd.read_excel('velviento.xlsx')
df_temperatura = pd.read_excel('temperatura.xlsx')
df_umidade = pd.read_excel('umidade.xlsx')

# Preparar una función para fusionar datos basados en columnas comunes
def merge_dataframes(base, other, suffix):
    return pd.merge(base, other, on=['BAIRRO', 'ANO', 'MES', 'DIA'], suffixes=('', suffix), how='inner')

# Crear un dataframe base a partir de aerosol que tiene menos datos
df_base = df_aerosol.copy()

# Fusionar cada uno de los otros dataframes con el base
df_base = merge_dataframes(df_base, df_temperatura, '_temp')
df_base = merge_dataframes(df_base, df_dirviento, '_dirviento')
df_base = merge_dataframes(df_base, df_velviento, '_velviento')
df_base = merge_dataframes(df_base, df_umidade, '_humedad')

# Limpiar columnas y renombrarlas para reflejar los datos correctos
df_base = df_base.rename(columns={
    'afternoon': 'AEROSOL_AFTERNOON',
    'midday': 'AEROSOL_MIDDAY',
    'afternoon_temp': 'TEMPERATURA_AFTERNOON',
    'midday_temp': 'TEMPERATURA_MIDDAY',
    'afternoon_dirviento': 'DIR_VIENTO_AFTERNOON',
    'midday_dirviento': 'DIR_VIENTO_MIDDAY',
    'afternoon_velviento': 'VEL_VIENTO_AFTERNOON',
    'midday_velviento': 'VEL_VIENTO_MIDDAY',
    'afternoon_humedad': 'HUMEDAD_AFTERNOON',
    'midday_humedad': 'HUMEDAD_MIDDAY'
})


columns_afternoon = ['BAIRRO', 'ANO', 'MES', 'DIA', 
                     'AEROSOL_AFTERNOON', 'TEMPERATURA_AFTERNOON', 
                     'DIR_VIENTO_AFTERNOON', 'VEL_VIENTO_AFTERNOON', 'HUMEDAD_AFTERNOON']

df_afternoon = df_base[columns_afternoon].dropna()

# Crear dataframe para 'midday'
columns_midday = ['BAIRRO', 'ANO', 'MES', 'DIA', 
                  'AEROSOL_MIDDAY', 'TEMPERATURA_MIDDAY', 
                  'DIR_VIENTO_MIDDAY', 'VEL_VIENTO_MIDDAY', 'HUMEDAD_MIDDAY']

df_midday = df_base[columns_midday].dropna()

# Ahora tienes dos dataframes: df_afternoon y df_midday
# Cada uno solo incluye las columnas para su respectivo período y solo filas completas

# Opcional: Guardar los resultados en archivos Excel
df_afternoon.to_excel('datos_afternoon.xlsx', index=False)
df_midday.to_excel('datos_midday.xlsx', index=False)


df_datos = pd.read_excel('datos.xlsx')
df_distancia = pd.read_excel('distancia.xlsx')

# Asegurarse de que los nombres de los barrios están en el mismo formato y corregir si es necesario
# Suponiendo que necesitamos renombrar 'nome' a 'BAIRRO' en df_distancia para que coincida con df_datos
df_distancia.rename(columns={'nome': 'BAIRRO'}, inplace=True)

# Realizar la fusión en base a la columna 'BAIRRO'
df_final = pd.merge(df_datos, df_distancia[['BAIRRO', 'Distance', 'latitud', 'longitud']], on='BAIRRO', how='left')

# Guardar el DataFrame resultante en un nuevo archivo Excel
df_final.to_excel('datos_actualizados.xlsx', index=False)
