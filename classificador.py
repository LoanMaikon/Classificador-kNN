import numpy as np
from collections import Counter
import csv
import os

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

# Caminho para o diretório das imagens
caminhoDiretorio = '/home/luan/Desktop/PKLot/PKLotSegmented'

# Retirando dados da UFPR04
caminhoUFPR04 = os.path.join(caminhoDiretorio, 'UFPR04')
caminhoArquivoCSV_treino_normalizado_UFPR04 = os.path.join(caminhoUFPR04, 'caracteristicas_treino_normalizado.csv')

# Definindo k-NN de 3 vizinhos
kValor = 3

# Vetores para armazenar os dados de treinamento
dadosTreinoUFPR04 = [] # X_treino
ocupacoesUFPR04 = [] # y_treino

# Preenchendo vetores de treinamento
with open(caminhoArquivoCSV_treino_normalizado_UFPR04, 'r') as arquivoCSV_treino_normalizado_UFPR04:
    leitor = csv.reader(arquivoCSV_treino_normalizado_UFPR04)
    
    # Iterando nas linhas do arquivo .csv para retirar características e ocupação
    for linha in leitor:
        # Separando os campos da linha
        campos = linha[0].split(';')

        # Colocando as características da linha em um vetor
        caracteristicas = np.array([float(campo) for campo in campos[:-1]])

        # Retirando a ocupação
        ocupacao = campos[-1]

        # Adicionando os dados de treino e a ocupação nas listas
        dadosTreinoUFPR04.append(caracteristicas)
        ocupacoesUFPR04.append(ocupacao)
    
# Convertendo listas para numpy arrays
dadosTreinoUFPR04 = np.array(dadosTreinoUFPR04)
ocupacoesUFPR04 = np.array(ocupacoesUFPR04)

# Diretório para .csv de teste
caminhoArquivoCSV_teste_UFPR04 = os.path.join(caminhoUFPR04, 'caracteristicas_teste.csv')

# Colocando as características de teste em um vetor
with open(caminhoArquivoCSV_teste_UFPR04, 'r') as arquivoCSV_teste_UFPR04:
    leitor = csv.reader(arquivoCSV_teste_UFPR04)

    caracteristicasTesteUFPR04 = []

    # Iterando nas linhas do arquivo .csv para retirar características
    for linha in leitor:
        # Separando os campos da linha
        campos = linha[0].split(';')

        # Colocando as características da linha em um vetor
        caracteristicas = np.array([float(campo) for campo in campos[:-1]])

        # Adicionando as características na lista
        caracteristicasTesteUFPR04.append(caracteristicas)

# Convertendo lista para numpy array
caracteristicasTesteUFPR04 = np.array(caracteristicasTesteUFPR04)

# Criar o classificador KNN
knn = KNN(kValor)
knn.fit(dadosTreinoUFPR04, ocupacoesUFPR04)

# Fazer previsões para as características de teste
previsoes = knn.predict(caracteristicasTesteUFPR04)

# Imprimir as previsões
for previsao in previsoes:
    # Printando previsão
    print('Previsão: ' + previsao)

    # Printando previsão certa
    print('Previsão certa: ' + ocupacoesUFPR04[previsao])