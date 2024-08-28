#importar livrarias
import os
import pandas as pd
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
from matplotlib.pyplot import subplots
from sklearn.pipeline import Pipeline
import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset
from torchmetrics import (MeanAbsoluteError , R2Score, MeanSquaredError)
from torchinfo import summary
from torchvision.io import read_image
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
torch.manual_seed(0)
from ISLP.torch import (SimpleDataModule , SimpleModule , ErrorTracker , rec_num_workers)

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

#inicializar o modelos
columns = ['DIRVI', 'VELVI', 'TEMP', 'UMI', 'aerosol', 'DIST_OCEAN']
best_r2 = -np.inf
best_predictors = None

# Probar todas as combinações posiveis da regressão
for r in range(1, len(columns) + 1):
    for predictors in combinations(columns, r):
        predictors = list(predictors)
        print(predictors)
        current_r2 = evaluate_model(no2_train, predictors)
        print(current_r2)
        
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


# Converter o array numpy estandarizado  para um DataFrame
Xs = pd.DataFrame(X_s, columns=X.columns, index=X.index)

#olhar quando tem mudança de bairro
bairro_changes = dados['BAIRRO'].ne(dados['BAIRRO'].shift()).cumsum()


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

# Dividir em treinamento teste e validação
X_train, X_test, y_train, y_test = train_test_split(sub_x[0], sub_y[0], test_size=0.2, random_state=42)

#X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#transformar df em np
X_train_np = X_train.to_numpy().astype(np.float32)
#X_val_np = X_val.to_numpy().astype(np.float32)
X_test_np = X_test.to_numpy().astype(np.float32)
y_train_np = y_train.to_numpy().astype(np.float32)
#y_val_np = y_val.to_numpy().astype(np.float32)
y_test_np = y_test.to_numpy().astype(np.float32)

#criar tensores
X_train_t = torch.tensor(X_train_np.astype(np.float32))
y_train_t = torch.tensor(y_train_np.astype(np.float32))
train = TensorDataset(X_train_t , y_train_t)

X_test_t = torch.tensor(X_test_np.astype(np.float32))
y_test_t = torch.tensor(y_test_np.astype(np.float32))
test = TensorDataset(X_test_t , y_test_t)

# Criando um SimpleDataModule
max_num_workers = rec_num_workers()
dm = SimpleDataModule(train ,
                           test ,
                           batch_size=32, # Tamanho dos lotes
                           num_workers=min(4, max_num_workers),
                           validation=0.1) # Conjunto de validação será 10% do tamanho do conjunto de treino


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

# Testando diferentes arquiteturas de rede

n_layers = 20

n_neuronios = [10, 50, 100, 200]

data = pd.DataFrame(columns=['Quantidade de Layers', 'Número de Neurônios', 'MSE', 'RMSE', 'R2'])

for n_neuronio in n_neuronios:
    for n in range(n_layers):
        # Criando a rede neural
        hidden_layers = np.full(n, n_neuronio)
        model = Network(X_s.shape[1], hidden_layers)
        
        # Definindo o modulo com a métrica RMSE
        module = SimpleModule.regression(model, metrics={'rmse': MeanSquaredError(squared=False)})
        
        # Objeto para salvar os arquivos logs
        logger = CSVLogger('logs', name='white_wine')
        
        # Treinando a rede
        n_epochs = 100
        trainer = Trainer(deterministic=True,
                          max_epochs=n_epochs, # Número de épocas
                          log_every_n_steps=5, # Número de passos em que serão salvas informações
                          logger=logger , # Logger em que serão salvas as informações
                          callbacks=[ErrorTracker()])
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
        filename = f"imagens/Neural_Network/RMSE_{n}_{n_neuronio}.png"        
        plt.savefig(filename)        
        
        # Coloque o modelo em modo de avaliação
        model.eval()

        # Faça as previsões com o modelo
        preds = model(X_test_t)
        
        # Converta os tensores para arrays NumPy para calcular o R²
        preds = preds.detach().cpu().numpy()
        
        mse = mean_squared_error(y_test,preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        data = data.append({'Quantidade de Layers': n, 'Número de Neurônios': n_neuronio, 'MSE': mse, 'RMSE': rmse, 'R2': r2}, ignore_index=True)
        
        del(model , trainer , module, logger)


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
