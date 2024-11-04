import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Ler excel
data = pd.read_excel('rf_sem_aerosol.xlsx')
#data = pd.read_excel('rf_com_aerosol.xlsx')

# Normalizar nome das colunas
data.columns = data.columns.str.lower().str.replace(' ', '_')

# Paso 0: Definir variables de entrada y salida (prediçao)

columnas_a_eliminar = ['pm25','no2']
X = data.drop(columnas_a_eliminar, axis=1)
y = data['pm25']

# Etapa 1: Dividir os dados em conjuntos de treinamento, validação e teste
# Conjunto de treinamento (X_train, y_train): usado para treinar o modelo.
# Conjunto de validação (X_val, y_val): usado para ajustar hiperparâmetros e evitar overfitting.
# Conjunto de testes (X_test, y_test): utilizado para avaliar o desempenho final do modelo.
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 50% para validación y 50% para prueba

#Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# etapa 2: Defina o espaço de pesquisa do hiperparâmetro
param_dist = {
    'n_estimators': [50, 100, 200, 300, 400, 500], # Aumentar el número de árboles
    'max_depth': [None, 10, 20, 30, 35], # Ampliar el rango de profundidad máxima
    'min_samples_split': [2, 5, 10, 15], # Más opciones para el número mínimo de muestras para dividir
    'min_samples_leaf': [1, 2, 3, 4, 5], # Más opciones para el número mínimo de muestras en una hoja
    'max_features': ['sqrt', 'log2', None], # Opción adicional para usar todas las características
    'bootstrap': [True, False], # Opción para utilizar el muestreo bootstrap
}

# etapa 3: Crear o modelo base
rf = RandomForestRegressor(random_state=42)

# etapa 4: Instanciar RandomizedSearchCV
# Experimentar diferentes combinações de hiperparâmetros aleatoriamente
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                   n_iter=50, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# etapa 5: Ajustar pesquisa aleatória ao conjunto de treinamento
random_search.fit(X_train_scaled, y_train)

# Obter os melhores parâmetros
# Mostrar os melhores hiperparâmetros encontrados durante a pesquisa aleatória.
best_params = random_search.best_params_
print(f"Mejores Parámetros: {best_params}")
# Imprimir el mejor puntaje (MSE negativo convertido a MSE positivo)
print("Mejor Score (MSE):", -random_search.best_score_)

# Paso 6: Evaluar el mejor modelo en el conjunto de validação
# Evalúa el rendimiento del mejor modelo en el conjunto de validación
best_model = random_search.best_estimator_
val_mse = mean_squared_error(y_val, best_model.predict(X_val_scaled))
val_r2 = r2_score(y_val, best_model.predict(X_val_scaled))
print(f'Mean Squared Error en el conjunto de validación: {val_mse:.2f}')
print(f'R^2 en el conjunto de validación: {val_r2:.2f}')

# Paso 7: Evaluar el modelo en el conjunto de teste
# Evalúa el rendimiento del modelo en el conjunto de prueba final
test_mse = mean_squared_error(y_test, best_model.predict(X_test_scaled))
test_r2 = r2_score(y_test, best_model.predict(X_test_scaled))
print(f'Mean Squared Error en el conjunto de prueba: {test_mse:.2f}')
print(f'R^2 en el conjunto de prueba: {test_r2:.2f}')

# Extraer la importancia de las características del mejor modelo
importances = best_model.feature_importances_

# Crear un dataframe con las características y su importancia
feature_importance = pd.DataFrame({'Variable': X_train_scaled.columns, 'Importance': importances})

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