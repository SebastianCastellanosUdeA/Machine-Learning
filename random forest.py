import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt

# Leer excel
#data = pd.read_excel('sem_aerosol.xlsx')
data = pd.read_excel('com_aerosol.xlsx')

# Normalizar el nombre de las columnas
data.columns = data.columns.str.lower().str.replace(' ', '_')


# Mostrar primeros 5 registros
data.head()


# Paso 0: Definir variables de entrada y salida (predicción)
columnas_a_eliminar = ['pm25','no2','ano']
X = data.drop(columnas_a_eliminar, axis=1)
y = data['pm25']

# Paso 1: Dividir los datos en conjuntos de entrenamiento, validación y prueba
# Conjunto de entrenamiento (X_train, y_train): usado para entrenar el modelo.
# Conjunto de validación (X_val, y_val): usado para ajustar los hiperparámetros y evitar el sobreajuste.
# Conjunto de prueba (X_test, y_test): usado para evaluar el rendimiento final del modelo.
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 50% para validación y 50% para prueba


# Paso 2: Definir el espacio de búsqueda de hiperparámetros
param_dist = {
    'n_estimators': [50, 100, 200, 300, 400, 500], # Aumentar el número de árboles
    'max_depth': [None, 10, 20, 30, 35], # Ampliar el rango de profundidad máxima
    'min_samples_split': [2, 5, 10, 15], # Más opciones para el número mínimo de muestras para dividir
    'min_samples_leaf': [1, 2, 3, 4, 5], # Más opciones para el número mínimo de muestras en una hoja
    'max_features': ['sqrt', 'log2', None], # Opción adicional para usar todas las características
    'bootstrap': [True, False], # Opción para utilizar el muestreo bootstrap
}

# Paso 3: Crear el modelo base
rf = RandomForestRegressor(random_state=42)

# Paso 4: Instanciar RandomizedSearchCV
# Prueba diferentes combinaciones de hiperparámetros de manera aleatoria
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                   n_iter=50, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Paso 5: Ajustar la búsqueda aleatoria al conjunto de entrenamiento
random_search.fit(X_train, y_train)

# Obtener los mejores parámetros
# Muestra los mejores hiperparámetros encontrados durante la búsqueda aleatoria.
best_params = random_search.best_params_
print(f"Mejores Parámetros: {best_params}")
# Imprimir el mejor puntaje (MSE negativo convertido a MSE positivo)
print("Mejor Score (MSE):", -random_search.best_score_)

# Paso 6: Evaluar el mejor modelo en el conjunto de validación
# Evalúa el rendimiento del mejor modelo en el conjunto de validación
best_model = random_search.best_estimator_
val_mse = mean_squared_error(y_val, best_model.predict(X_val))
val_r2 = r2_score(y_val, best_model.predict(X_val))
print(f'Mean Squared Error en el conjunto de validación: {val_mse:.2f}')
print(f'R^2 en el conjunto de validación: {val_r2:.2f}')

# Paso 7: Evaluar el modelo en el conjunto de prueba
# Evalúa el rendimiento del modelo en el conjunto de prueba final
test_mse = mean_squared_error(y_test, best_model.predict(X_test))
test_r2 = r2_score(y_test, best_model.predict(X_test))
print(f'Mean Squared Error en el conjunto de prueba: {test_mse:.2f}')
print(f'R^2 en el conjunto de prueba: {test_r2:.2f}')

# Extraer la importancia de las características del mejor modelo
importances = best_model.feature_importances_

# Crear un dataframe con las características y su importancia
feature_importance = pd.DataFrame({'Variable': X_train.columns, 'Importance': importances})

# Ordenar las características según su importancia de mayor a menor
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)


# Graficar las importancias
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Variable'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.ylabel('Variables')
plt.title('Importance Random Forest features')
plt.gca().invert_yaxis()  # Invertir el eje Y para que la característica más importante esté en la parte superior
plt.show()






# # Imprimir los mejores parámetros
# print("Mejores Parámetros:", random_search.best_params_)

# # Imprimir el mejor puntaje (MSE negativo convertido a MSE positivo)
# print("Mejor Score (MSE):", -random_search.best_score_)

# # Imprimir todos los resultados detallados
# results = random_search.cv_results_
# for i in range(len(results['params'])):
#     print(f"Parámetros: {results['params'][i]}, Media MSE: {-results['mean_test_score'][i]:.2f}, Desviación estándar: {results['std_test_score'][i]:.2f}")



y_pred = best_model.predict(X_test)
results_df = pd.DataFrame({'Real': y_test, 'Predicho': y_pred})

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Real')
plt.plot(y_pred, label='Predicho', alpha=0.7)
plt.legend()
plt.title('Valores Reales vs Predichos')
plt.xlabel('Índice')
plt.ylabel('PM2.5')
plt.show()