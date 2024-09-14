# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 00:00:12 2024

@author: sebas
"""

#importar livrarias
import os
import pandas as pd
import copy
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl
import numpy as np
from itertools import combinations
import sklearn.model_selection as skm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from matplotlib.pyplot import subplots
from sklearn.pipeline import Pipeline
import torch
from torch import nn
from torch.optim import RMSprop, Adam
from torch.utils.data import TensorDataset
from torchmetrics import (MeanAbsoluteError , R2Score, MeanSquaredError)
from torchinfo import summary
from torchvision.io import read_image
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
torch.manual_seed(0)
from ISLP.torch import (SimpleDataModule , SimpleModule , ErrorTracker , rec_num_workers)

#criar funções

#função para criar modelo de regrresão
def evaluate_model(X_train, predictors):
    model = LinearRegression()
    model.fit(X_train[predictors], X_train['NO2'])
    predictions = model.predict(X_train[predictors])
    return r2_score(X_train['NO2'], predictions)

#função para predecir dados faltantes
def predict_missing_values(X_train, X_missing, best_predictors):
    model = LinearRegression()
    model.fit(X_train[best_predictors], X_train['NO2'])
    predicted_values = model.predict(X_missing[best_predictors])
    return predicted_values

#rede neural
class Network(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()
        
        # Crie uma lista para armazenar as camadas da rede
        layers = []
        in_features = input_size
        
        for out_features in hidden_layers:
            # Adicione uma camada linear seguida de uma função de ativação ReLU
            layers.append(nn.Linear(in_features, out_features))
            #layers.append(nn.Sigmoid())
            layers.append(nn.ReLU())  
            in_features = out_features  # Atualize o número de neurônios de entrada para a próxima camada
        
        # Adicione a camada de saída
        layers.append(nn.Linear(in_features, 1))
        
        # Crie a sequência de camadas
        self.sequential = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.flatten(x)
        return torch.flatten(self.sequential(x))
    
    
# Função para salvar o RMSE durante o treinamento
def summary_plot(results, ax, col='loss', valid_legend='Validation', training_legend='Training', ylabel='Loss', fontsize=20):
    for (column, color, label) in zip([f'train_{col}_epoch', f'valid_{col}'], ['black', 'red'], [training_legend, valid_legend]):
        ax.plot(results['epoch'], results[column], label=label, marker='o', color=color)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax

# Leer os dados
dados = pd.read_excel("input.xlsx")
# Seleccionar as colunas de entrada (features) y a coluna de saida (target)
X = dados[['DIRVI', 'VELVI', 'TEMP', 'UMI', 'aerosol','NO2','DIST_OCEAN']]
y = dados['PM25']

#assegurar que NO2 é númerico
X['NO2'] = pd.to_numeric(X['NO2'], errors='coerce')

'''
Como realizou-se o preenchumento de dados faltantes:
dir vento - colocou-se média da estaçao
vel vento - média no mesmo ano
'''
#olhar os dados faltantes
no2_train = X.dropna(subset=['NO2'])
no2_missing = X[X['NO2'].isnull()]

#inicializar o modelos
columns = ['DIRVI', 'VELVI', 'TEMP', 'UMI', 'aerosol', 'DIST_OCEAN']
best_r2 = -np.inf
best_predictors = None

# Probar todas as combinações posiveis da regressão
for r in range(1, len(columns) + 1):
    for predictors in combinations(columns, r):
        predictors = list(predictors)
        current_r2 = evaluate_model(no2_train, predictors)
        
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_predictors = predictors

# Usar o melhor modelo para predecir e imputar os valores faltantes
predicted_values = predict_missing_values(no2_train, no2_missing, best_predictors)

# Imputar os valores no DataFrame original
X.loc[X['NO2'].isnull(), 'NO2'] = predicted_values

# estandar
scaler = StandardScaler()

# Ajustar y transformar os dados
X_s = scaler.fit_transform(X)

joblib.dump(scaler, 'scaler.pkl')

# Converter o array numpy estandarizado  para um DataFrame
Xs = pd.DataFrame(X_s, columns=X.columns, index=X.index)

#optimizar o número de neuronios 
best_num_layers = None
best_num_neurons = None
best_rmse = np.inf

X_trn0, X_tst0, y_trn0, y_tst0 = train_test_split(Xs, y, test_size=0.3, random_state=42)

#transformar df em np
Xs_train = X_trn0.to_numpy().astype(np.float32)
Xs_test = X_tst0.to_numpy().astype(np.float32)
y_train = y_trn0.to_numpy().astype(np.float32)
y_test = y_tst0.to_numpy().astype(np.float32)

#criar tensores
X_train_t = torch.tensor(Xs_train.astype(np.float32))
y_train_t = torch.tensor(y_train.astype(np.float32))
train = TensorDataset(X_train_t , y_train_t)

X_test_t = torch.tensor(Xs_test.astype(np.float32))
y_test_t = torch.tensor(y_test.astype(np.float32))
test = TensorDataset(X_test_t , y_test_t)

# Criando um SimpleDataModule
max_num_workers = rec_num_workers()
dm = SimpleDataModule(train,
                      test,
                      batch_size=32, # Tamanho dos lotes
                      num_workers=min(4, max_num_workers),
                      validation=0.2) # Conjunto de validação será 20% do tamanho do conjunto de treino
 
# Testando diferentes arquiteturas de rede
    
n_layers = 20

n_neuronios = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]

data = pd.DataFrame(columns=['subconjunto','Folder','Quantidade de Layers', 'Número de Neurônios', 'MSE', 'RMSE', 'R2'])

for n_neuronio in n_neuronios:
    for n in range(n_layers):
        # Criando a rede neural
        hidden_layers = np.full(n, n_neuronio)
        model = Network(Xs_train.shape[1], hidden_layers)
        
        #optimizer
        optimizer = Adam(model.parameters())
        # Definindo o modulo com a métrica RMSE
        module = SimpleModule.regression(model,optimizer=optimizer, metrics={'rmse': MeanSquaredError(squared=False)})
        
        # Objeto para salvar os arquivos logs
        logger = CSVLogger('logs', name='particulate_matter')
        
        # Definindo o critério de parada temprana baseado no RMSE
        early_stopping = EarlyStopping(
            monitor='valid_rmse',
            min_delta=0.0001,  # Diferencia mínima para considerar una melhoria
            patience=20,  # Número de épocas sem melhorar antes de parar
            verbose=True,
            mode='min'  # Queremos minimizar o RMSE
        )
        # Treinando a rede
        n_epochs = 100
        trainer = Trainer(deterministic=True,
                          max_epochs=n_epochs, # Número de épocas
                          log_every_n_steps=5, # Número de passos em que serão salvas informações
                          logger=logger , # Logger em que serão salvas as informações
                          callbacks=[ErrorTracker(), early_stopping])
        trainer.fit(module , datamodule=dm)
        
        # Avaliando a performance do modelo para o conjunto de teste
        trainer.test(module , datamodule=dm)
        
        # Criando um plot de MAE (mean absolute error) em função do número de épocas
        results = pd.read_csv(logger.experiment.metrics_file_path)
        
        plt.clf()
        fig, ax = subplots(1, 1, figsize=(6, 6))
        ax = summary_plot(results,
                        ax,
                        col='rmse',
                        ylabel='RMSE',
                        valid_legend='Validation (=Test)')
        #ax.set_ylim([0, 400])
        ax.set_xticks(np.linspace(0, n_epochs, 11).astype(int));
        #filename = f"imagens/Neural_Network/RMSE_{n}_{n_neuronio}.png"        
        #plt.savefig(filename)        
        
        # Coloque o modelo em modo de avaliação
        model.eval()

        # Faça as previsões com o modelo
        preds = model(X_test_t)
        
        # Converta os tensores para arrays NumPy para calcular o R²
        preds = preds.detach().cpu().numpy()
        
        mse = mean_squared_error(y_test,preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        new_row = pd.DataFrame({
                                'subconjunto': ['inicial'],
                                'Folder': ['inicial'],
                                'Quantidade de Layers': [n],
                                'Número de Neurônios': [n_neuronio],
                                'MSE': [mse],
                                'RMSE': [rmse],
                                'R2': [r2]
                            })

        data = pd.concat([data, new_row], ignore_index=True)
        
        # Actualizar la mejor configuración si el RMSE es menor
        if rmse < best_rmse:
            best_rmse = rmse
            best_num_layers = n
            best_num_neurons = n_neuronio
            previous_model_state = model.state_dict()
            
        del(model , trainer , module, logger)

previous_model_state = None
# Configure o estilo dos gráficos e use a paleta personalizada
sns.set(style="whitegrid", palette=sns.color_palette("tab10"))

# Crie o gráfico de linhas com as séries de RMSE
plt.figure(figsize=(10, 6))  # Ajuste o tamanho da figura conforme necessário

# Use o seaborn para criar o gráfico de linhas e ajuste a transparência (alpha) para 1
sns.lineplot(x="Quantidade de Layers", y="RMSE", hue="Número de Neurônios", data=data, marker="o", palette=sns.color_palette("tab10"), alpha=1)

# Adicione um título ao gráfico
plt.title("RMSE por Número de Neurônios e Quantidade de Layers")

# Mostre o gráfico
plt.show()

#Criar modelo 
n_splits = 5

#treinamento por localização

#olhar quando tem mudança de bairro, FÁTIMA E MOURA BRASIL TEM POUCOS DADOS
dados_modificado = dados['BAIRRO'].replace({'FÁTIMA': 'IGNORAR', 'MOURA BRASIL': 'IGNORAR'})

bairro_changes = dados_modificado.ne(dados_modificado.shift()).cumsum()

sub_x = []

# Agrupar pelo identificador de cambio de bairro
for _, group in Xs.groupby(bairro_changes):
    subset = group[['DIRVI', 'VELVI', 'TEMP', 'UMI', 'aerosol','NO2','DIST_OCEAN']]
    sub_x.append(subset)

sub_y = []

# Agrupar y pelo identificador de cambio de bairro
for _, group in dados.groupby(bairro_changes):
    start_idx = group.index[0]
    end_idx = group.index[-1] + 1  # Adicionar 1 para incluir o último índice na divisão
    subset_y = y.iloc[start_idx:end_idx]
    sub_y.append(subset_y)
    
# variaveis acumulativas
accumulated_x = sub_x[0]
accumulated_y = sub_y[0]

print(f"Usando la mejor configuración: {best_num_layers} layers, {best_num_neurons} neurônios")

for i in range(len(sub_x)):  # Empezamos desde sub_x[0] y sub_y[0]
    if i > 0:
        print(f"Iteración {i}")
        # Actualizar los conjuntos de datos acumulados
        accumulated_x = pd.concat([accumulated_x, sub_x[i]], axis=0)
        accumulated_y = pd.concat([accumulated_y, sub_y[i]], axis=0)
    else:
        print(f"Iteración {i}")
        
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []
    fold_test_indices = {}
    
    for fold, (train_index, test_index) in enumerate(kf.split(accumulated_x)):
        print(f"Fold {fold + 1}")
        
        # Dividir el conjunto de datos en entrenamiento y test según las particiones de KFold
        X_trn, X_tst = accumulated_x.iloc[train_index], accumulated_x.iloc[test_index]
        y_trn, y_tst = accumulated_y.iloc[train_index], accumulated_y.iloc[test_index]
        fold_test_indices[fold] = test_index 
    
        #transformar df em np
        Xs_train = X_trn.to_numpy().astype(np.float32)
        Xs_test = X_tst.to_numpy().astype(np.float32)
        y_train = y_trn.to_numpy().astype(np.float32)
        y_test = y_tst.to_numpy().astype(np.float32)
        
        #criar tensores
        X_train_t = torch.tensor(Xs_train.astype(np.float32))
        y_train_t = torch.tensor(y_train.astype(np.float32))
        train = TensorDataset(X_train_t , y_train_t)
        
        X_test_t = torch.tensor(Xs_test.astype(np.float32))
        y_test_t = torch.tensor(y_test.astype(np.float32))
        test = TensorDataset(X_test_t , y_test_t)
        
        # Criando um SimpleDataModule
        max_num_workers = rec_num_workers()
        dm = SimpleDataModule(train,
                              test,
                              batch_size=32, # Tamanho dos lotes
                              num_workers=min(4, max_num_workers),
                              validation=0.2) # Conjunto de validação será 20% do tamanho do conjunto de treino
     
        hidden_layers = np.full(best_num_layers, best_num_neurons)
        model = Network(Xs_train.shape[1], hidden_layers)
        
        #optimizer
        optimizer = Adam(model.parameters())

        # Cargar el estado del modelo del fold anterior
        if previous_model_state:
            model.load_state_dict(previous_model_state)
        
        # Definindo o modulo com a métrica RMSE
        module = SimpleModule.regression(model,optimizer=optimizer, metrics={'rmse': MeanSquaredError(squared=False)})
        
        # Objeto para salvar os arquivos logs
        logger = CSVLogger('logs', name='particulate_matter')
        
        # Definindo o critério de parada temprana baseado no RMSE
        early_stopping = EarlyStopping(
            monitor='valid_rmse',
            patience=20,  # Número de épocas sem melhorar antes de parar
            min_delta=0.0001,  # Diferencia mínima para considerar una melhoria
            verbose=True,
            mode='min'  # Queremos minimizar o RMSE
        )
        
        # Treinando a rede
        n_epochs = 100
        trainer = Trainer(deterministic=True,
                          max_epochs=n_epochs, # Número de épocas
                          log_every_n_steps=5, # Número de passos em que serão salvas informações
                          logger=logger , # Logger em que serão salvas as informações
                          callbacks=[ErrorTracker(), early_stopping])
        trainer.fit(module , datamodule=dm)
        
        # Avaliando a performance do modelo para o conjunto de teste
        trainer.test(module , datamodule=dm)    
        
        # Coloque o modelo em modo de avaliação
        model.eval()

        # Faça as previsões com o modelo
        preds = model(X_test_t)
        
        # Converta os tensores para arrays NumPy para calcular o R²
        preds = preds.detach().cpu().numpy()
        
        mse = mean_squared_error(y_test,preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        rmse_scores.append(rmse)
        
        new_row = pd.DataFrame({
                                'subconjunto': [i + 1],
                                'Folder': [fold + 1],
                                'Quantidade de Layers': [n],
                                'Número de Neurônios': [n_neuronio],
                                'MSE': [mse],
                                'RMSE': [rmse],
                                'R2': [r2]
                            })

        data = pd.concat([data, new_row], ignore_index=True)
        
        del(model, trainer, module, logger)
    
    # Convertir la lista de RMSEs en un array de NumPy
    rmse_scores = np.array(rmse_scores)
    # Calcular la media y la desviación estándar de los RMSE
    mean_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()

    # Definir un umbral (2 desviaciones estándar)
    threshold = mean_rmse +  2 * std_rmse
    # Filtrar los folds que estén dentro del rango aceptable
    atipic = np.where(rmse_scores >= threshold)[0]

    # Crear una lista para almacenar todos los índices atípicos encontrados
    atypical_indices = []

    # Recorrer los números de folds que están en `atipic`
    for fold in atipic:
        # Revisar en el diccionario `fold_test_indices` los índices de test para el fold atípico
        test_indices = fold_test_indices[fold]  
        atypical_indices.extend(test_indices)  # Añadir estos índices a la lista de índices atípicos

    at_idx = list(atypical_indices)

    filas_a_agregar = []
    valores_a_agregar = []

    # Recorrer todos los índices de X_train
    for idx in range(len(accumulated_x)):
        # Si el índice no está en atypical_indices, agregar la fila a X_train_f
        if idx not in atypical_indices:
            filas_a_agregar.append(accumulated_x.iloc[idx])
            valores_a_agregar.append(accumulated_y.iloc[idx])
            
    #MODELO FINAL por LOCALIZAÇÃO
    X_train_f = pd.DataFrame(filas_a_agregar, columns=accumulated_x.columns)
    y_train_f = pd.Series(valores_a_agregar, name=accumulated_y.name)
    
    accumulated_x = copy.deepcopy(X_train_f)
    accumulated_y = copy.deepcopy(y_train_f)
    
    Xs_train = X_train_f.to_numpy().astype(np.float32)
    Xs_test = X_tst0.to_numpy().astype(np.float32)
    y_train = y_train_f.to_numpy().astype(np.float32)
    y_test = y_tst0.to_numpy().astype(np.float32)

    #criar tensores
    X_train_t = torch.tensor(Xs_train.astype(np.float32))
    y_train_t = torch.tensor(y_train.astype(np.float32))
    train = TensorDataset(X_train_t , y_train_t)

    X_test_t = torch.tensor(Xs_test.astype(np.float32))
    y_test_t = torch.tensor(y_test.astype(np.float32))
    test = TensorDataset(X_test_t , y_test_t)

    max_num_workers = rec_num_workers()
    dm1 = SimpleDataModule(train,
                          test,
                          batch_size=32, # Tamanho dos lotes
                          num_workers=min(4, max_num_workers),
                          validation=0.2) # Conjunto de validação será 20% do tamanho do conjunto de treino

    hidden_layers = np.full(best_num_layers, best_num_neurons)
    model_f = Network(Xs_train.shape[1], hidden_layers)

    #optimizer
    optimizer = Adam(model_f.parameters())
    
    # Cargar el estado del modelo del fold anterior
    if previous_model_state:
        model_f.load_state_dict(previous_model_state)

    # Definindo o modulo com a métrica RMSE
    module = SimpleModule.regression(model_f,optimizer=optimizer, metrics={'rmse': MeanSquaredError(squared=False)})

    # Objeto para salvar os arquivos logs
    logger = CSVLogger('logs', name='particulate_matter')

    # Definindo o critério de parada temprana baseado no RMSE
    early_stopping = EarlyStopping(
        monitor='valid_rmse',
        patience=20,  # Número de épocas sem melhorar antes de parar
        min_delta=0.0001,  # Diferencia mínima para considerar una melhoria
        verbose=True,
        mode='min'  # Queremos minimizar o RMSE
    )

    # Treinando a rede
    n_epochs = 200
    trainer = Trainer(deterministic=True,
                      max_epochs=n_epochs, # Número de épocas
                      log_every_n_steps=5, # Número de passos em que serão salvas informações
                      logger=logger , # Logger em que serão salvas as informações
                      callbacks=[ErrorTracker(), early_stopping])
    trainer.fit(module , datamodule=dm1)

    # Avaliando a performance do modelo para o conjunto de teste
    trainer.test(module , datamodule=dm1)

    # Coloque o modelo em modo de avaliação
    model_f.eval()

    # Faça as previsões com o modelo
    preds = model_f(X_test_t)

    # Converta os tensores para arrays NumPy para calcular o R²
    preds = preds.detach().cpu().numpy()

    # Criando um plot de MAE (mean absolute error) em função do número de épocas
    results2 = pd.read_csv(logger.experiment.metrics_file_path)

    mse = mean_squared_error(y_test,preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    plt.clf()
    fig, ax = subplots(1, 1, figsize=(6, 6))
    ax = summary_plot(results,
                    ax,
                    col='rmse',
                    ylabel='RMSE',
                    valid_legend='Validation (=Test)')
    #ax.set_ylim([0, 400])
    ax.set_xticks(np.linspace(0, n_epochs, 11).astype(int));
    #filename = f"imagens/Neural_Network/RMSE_{n}_{n_neuronio}.png"        
    #plt.savefig(filename)        

    new_row = pd.DataFrame({
                            'subconjunto': [i + 1],
                            'Folder': 'def',
                            'Quantidade de Layers': [best_num_layers],
                            'Número de Neurônios': [best_num_neurons],
                            'MSE': [mse],
                            'RMSE': [rmse],
                            'R2': [r2]
                        })

    data = pd.concat([data, new_row], ignore_index=True)
    
    if model_f.state_dict():
        previous_model_state = model_f.state_dict()
    
    if i < (len(sub_x) - 1):
        del(model_f, trainer, module, logger)
      

    
# Cargar el scaler previamente guardado
scaler = joblib.load('scaler.pkl')

# Normalizar los nuevos datos usando el scaler ajustado
X_new_scaled = scaler.transform(X)