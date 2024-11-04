# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:15:16 2024

@author: Trama  Sebas
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

'''
dadosi = pd.read_excel("ml_sem_aerosol.xlsx")
dadosi['NO2'] = pd.to_numeric(dadosi['NO2'], errors='coerce')
# Eliminar las filas donde 'NO2' es NaN
dados = dadosi.dropna(subset=['NO2'])
# Reiniciar los índices
dados = dados.reset_index(drop=True)
'''
# Leer os dados
dados = pd.read_excel("ml_sem_aerosol.xlsx")
dados['NO2'] = pd.to_numeric(dados['NO2'], errors='coerce')

#treinamento por localização

#olhar quando tem mudança de bairro, FÁTIMA E MOURA BRASIL TEM POUCOS DADOS
dados_modificado = dados['BAIRRO'].replace({'FÁTIMA': 'IGNORAR', 'MOURA BRASIL': 'IGNORAR'})

bairro_changes = dados_modificado.ne(dados_modificado.shift()).cumsum()
dados['bairro_changes'] = bairro_changes

# Seleccionar as colunas de entrada (features) y a coluna de saida (target)
y = dados['PM25']

#escolher as variaveis
X = dados[['ANO','DIRVI', 'UMI','VELVI','DIST_OCEAN', 'NO2','TEMP']]

# estandar
scaler = StandardScaler()

# Ajustar y transformar os dados
X_s = scaler.fit_transform(X)

joblib.dump(scaler, 'scaler.pkl')

# Converter o array numpy estandarizado  para um DataFrame
Xs = pd.DataFrame(X_s, columns=X.columns, index=X.index)

X_trn0, X_tst0, y_trn0, y_tst0 = train_test_split(Xs, y, test_size=0.2, random_state=42)

# Identificar los cambios de barrio en el conjunto de entrenamiento
bairro_changes_train = bairro_changes.loc[X_trn0.index]

bairro_changes_trn = bairro_changes_train.sort_index()
X_sorted = X_trn0.sort_index()
y_sorted = y_trn0.sort_index()


#optimizar o número de neuronios 
best_num_layers = None
best_num_neurons = None
best_rmse = np.inf
best_r2 = -np.inf


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
 
# Testando diferentes arquiteturas de rede e escolher a melhor
    
n_layers = 20

n_neuronios = [1, 10, 20, 30, 50, 100, 200]


data = pd.DataFrame(columns=['subconjunto','Folder','Quantidade de Layers', 'Número de Neurônios', 'MSE', 'RMSE', 'R2'])

for n_neuronio in n_neuronios:
    for n in range(n_layers):
        # Criando a rede neural
        hidden_layers = np.full(n, n_neuronio)
        model = Network(Xs_train.shape[1], hidden_layers)
        
        #optimizer
        optimizer = Adam(model.parameters())
        # Definindo o modulo com a métrica RMSE
        module = SimpleModule.regression(model,optimizer=optimizer,
                                         metrics={'rmse': MeanSquaredError(squared=False)})
        
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
        
        # Actualizar la mejor configuración si el RMSE es menor y guardar
        if rmse < best_rmse:
            best_rmse = rmse
            best_num_layers = n
            best_num_neurons = n_neuronio
            best_r2 = r2
            previous_model_state = model.state_dict()
            model_f = copy.deepcopy(model)
            optimizer_f = copy.deepcopy(model)
            
        del(model , trainer , module, logger)

      
# Guardar el estado del modelo junto con el optimizador
checkpoint = {
    'model_state_dict': model_f.state_dict(),
    'optimizer_state_dict': optimizer_f.state_dict(),
    'epoch': n_epochs,  
}

torch.save(checkpoint, 'modelo_entrenado_completo.pth')

#Cargar los nuevos datos
nuevos_datos = pd.read_excel("nuevos.xlsx")

# Cargar el escalador usado en el entrenamiento
scaler = joblib.load('scaler.pkl')
nuevos_datos_escalados = scaler.transform(nuevos_datos)

# Convertir los datos a tensores
nuevos_datos_t = torch.tensor(nuevos_datos_escalados.astype(np.float32))

#  Cargar el modelo entrenado
model_f = Network(nuevos_datos_t.shape[1], hidden_layers)
optimizer = Adam(model_f.parameters())
checkpoint = torch.load('modelo_entrenado_completo.pth')
model_f.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model_f.eval()  # Poner el modelo en modo de evaluación

# Hacer las predicciones
with torch.no_grad():  # Desactivar el cálculo del gradiente
    predicciones = model_f(nuevos_datos_t)

#Convertir las predicciones a un formato usable
predicciones = predicciones.detach().cpu().numpy()

# Mostrar las predicciones
print(predicciones)

# Guardar las predicciones en un archivo CSV si es necesario
predicciones_df = pd.DataFrame(predicciones, columns=['Prediccion_MP2.5'])
predicciones_df.to_excel('predicciones_nuevos_datos.xlsx', index=False)
