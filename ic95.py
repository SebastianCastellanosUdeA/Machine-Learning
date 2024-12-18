import pandas as pd
import numpy as np
from scipy import stats

# Cargar los datos desde un archivo Excel
df = pd.read_excel('datos_projetados_con_pm25.xlsx')

# Agrupar los datos por 'bairro' y 'ano'
grouped = df.groupby(['bairro', 'ano'])

# Función para calcular el intervalo de confianza del 95%
def conf_interval(data):
    ci = stats.norm.interval(0.95, loc=np.mean(data), scale=stats.sem(data))
    return pd.Series({'mean': np.mean(data), 'ci95_low': ci[0], 'ci95_high': ci[1]})

# Aplicar la función a cada grupo para calcular el promedio e intervalos de confianza
results = grouped['pm25_predicted'].apply(conf_interval).reset_index()

# Pivotar el dataframe para reorganizar los datos
pivot_df = results.pivot(index=['bairro', 'ano'], columns='level_2', values='pm25_predicted')

# Ahora, pivot_df debería tener 'mean', 'ci95_high' y 'ci95_low' como columnas
# y 'bairro' y 'ano' como índice. Usamos reset_index para hacer 'bairro' y 'ano' columnas nuevamente.
pivot_df = pivot_df.reset_index()

# Renombrar las columnas para claridad
pivot_df.columns = ['bairro', 'ano', 'mean', 'ci95_high', 'ci95_low']

# Ahora pivot_df es tu dataframe transformado
print(pivot_df)

pivot_df.to_excel('calculo_pm25.xlsx', index=False)
