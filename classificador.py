import numpy as np
from collections import Counter
import csv
import os

def distanciaEuclidiana(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_treino = X
        self.y_treino = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # Distancias entre o ponto x e todos os outros pontos
        distancias = [distanciaEuclidiana(x, x_treino) for x_treino in self.X_treino]
    
        # Pegando os índices dos k vizinhos mais próximos
        k_indices = np.argsort(distancias)[:self.k]
        ocupacoesProximas = [self.y_treino[i] for i in k_indices]

        # Maior ocorrência de ocupação entre os k vizinhos mais próximos
        predicao = Counter(ocupacoesProximas).most_common()
        return predicao[0][0]

# Gera vetor de dados para treino e teste
def gerarVetorDados(caminhoArquivoCSV):
    # Vetores para armazenar os dados de treinamento
    dadosTreino = [] # X_treino
    ocupacoesTreino = [] # y_treino

    # Preenchendo vetores de treinamento
    with open(caminhoArquivoCSV, 'r') as arquivoCSV_treino_normalizado:
        leitor = csv.reader(arquivoCSV_treino_normalizado)
        
        # Iterando nas linhas do arquivo .csv para retirar características e ocupação
        for linha in leitor:
            # Separando os campos da linha
            campos = linha[0].split(';')

            # Colocando as características da linha em um vetor
            caracteristicas = np.array([float(campo) for campo in campos[:-1]])

            # Retirando a ocupação
            ocupacao = campos[-1]

            # Adicionando os dados de treino e a ocupação nas listas
            dadosTreino.append(caracteristicas)
            ocupacoesTreino.append(ocupacao)
    
    # Convertendo listas para numpy arrays
    dadosTreino = np.array(dadosTreino)
    ocupacoesTreino = np.array(ocupacoesTreino)

    return dadosTreino, ocupacoesTreino

# Caminho para o diretório das imagens
caminhoDiretorio = '/home/luan/Desktop/PKLot/PKLotSegmented'

# Caminhos para as pastas das universidades
caminhoUFPR04 = os.path.join(caminhoDiretorio, 'UFPR04')
caminhoPUC = os.path.join(caminhoDiretorio, 'PUC')
caminhoUFPR05 = os.path.join(caminhoDiretorio, 'UFPR05')

# Diretório para .csv de treino
caminhoArquivoCSV_treino_normalizado_UFPR04 = os.path.join(caminhoUFPR04, 'caracteristicas_treino_normalizado.csv')
caminhoArquivoCSV_treino_normalizado_PUC = os.path.join(caminhoPUC, 'caracteristicas_treino_normalizado.csv')
caminhoArquivoCSV_treino_normalizado_UFPR05 = os.path.join(caminhoUFPR05, 'caracteristicas_treino_normalizado.csv')

# Diretório para .csv de teste
caminhoArquivoCSV_teste_normalizado_UFPR04 = os.path.join(caminhoUFPR04, 'caracteristicas_teste_normalizado.csv')
caminhoArquivoCSV_teste_normalizado_PUC = os.path.join(caminhoPUC, 'caracteristicas_teste_normalizado.csv')
caminhoArquivoCSV_teste_normalizado_UFPR05 = os.path.join(caminhoUFPR05, 'caracteristicas_teste_normalizado.csv')

# Definindo k-NN de 3 vizinhos
kValor = 3

dadosTreinoUFPR04, ocupacoesTreinoUFPR04 = gerarVetorDados(caminhoArquivoCSV_treino_normalizado_UFPR04)
dadosTreinoPUC, ocupacoesTreinoPUC = gerarVetorDados(caminhoArquivoCSV_treino_normalizado_PUC)
dadosTreinoUFPR05, ocupacoesTreinoUFPR05 = gerarVetorDados(caminhoArquivoCSV_treino_normalizado_UFPR05)

# Vetores para armazenar os dados de teste
dadosTesteUFPR04, ocupacoesTesteUFPR04 = gerarVetorDados(caminhoArquivoCSV_teste_normalizado_UFPR04)
dadosTestePUC, ocupacoesTestePUC = gerarVetorDados(caminhoArquivoCSV_teste_normalizado_PUC)
dadosTesteUFPR05, ocupacoesTesteUFPR05 = gerarVetorDados(caminhoArquivoCSV_teste_normalizado_UFPR05)

knn = KNN(kValor) # Inicializando k = 3

# Criando o classificador KNN para UFPR04
knn.fit(dadosTreinoUFPR04, ocupacoesTreinoUFPR04)

# Fazer previsões para as características de teste
previsoesUFPR04xUFPR04 = knn.predict(caracteristicasTesteUFPR04)
previsoesUFPR04xPUC = knn.predict(caracteristicasTestePUC)
previsoesUFPR04xUFPR05 = knn.predict(caracteristicasTesteUFPR05)

# Imprimir acurácia
acuraciaUFPR04xUFPR04 = np.sum(previsoesUFPR04xUFPR04 == ocupacoesTesteUFPR04) / len(ocupacoesTesteUFPR04)
acuraciaUFPR04xPUC = np.sum(previsoesUFPR04xPUC == ocupacoesTestePUC) / len(ocupacoesTestePUC)
acuraciaUFPR04xUFPR05 = np.sum(previsoesUFPR04xUFPR05 == ocupacoesTesteUFPR05) / len(ocupacoesTesteUFPR05)

print("Acurácia (treino) UFPR04 x (teste) UFPR04: ", acuraciaUFPR04xUFPR04)
print("Acurácia (treino) UFPR04 x (teste) PUC: ", acuraciaUFPR04xPUC)
print("Acurácia (treino) UFPR04 x (teste) UFPR05: ", acuraciaUFPR04xUFPR05)

# Criando o classificador KNN para PUC
knn.fit(dadosTreinoPUC, ocupacoesTreinoPUC)

# Fazer previsões para as características de teste
previsoesPUCxUFPR04 = knn.predict(caracteristicasTesteUFPR04)
previsoesPUCxPUC = knn.predict(caracteristicasTestePUC)
previsoesPUCxUFPR05 = knn.predict(caracteristicasTesteUFPR05)

# Imprimir acurácia
acuraciaPUCxUFPR04 = np.sum(previsoesPUCxUFPR04 == ocupacoesTesteUFPR04) / len(ocupacoesTesteUFPR04)
acuraciaPUCxPUC = np.sum(previsoesPUCxPUC == ocupacoesTestePUC) / len(ocupacoesTestePUC)
acuraciaPUCxUFPR05 = np.sum(previsoesPUCxUFPR05 == ocupacoesTesteUFPR05) / len(ocupacoesTesteUFPR05)

print("Acurácia (treino) PUC x (teste) UFPR04: ", acuraciaPUCxUFPR04)
print("Acurácia (treino) PUC x (teste) PUC: ", acuraciaPUCxPUC)
print("Acurácia (treino) PUC x (teste) UFPR05: ", acuraciaPUCxUFPR05)

# Criando o classificador KNN para UFPR05
knn.fit(dadosTreinoUFPR05, ocupacoesTreinoUFPR05)

# Fazer previsões para as características de teste
previsoesUFPR05xUFPR04 = knn.predict(caracteristicasTesteUFPR04)
previsoesUFPR05xPUC = knn.predict(caracteristicasTestePUC)
previsoesUFPR05xUFPR05 = knn.predict(caracteristicasTesteUFPR05)

# Imprimir acurácia
acuraciaUFPR05xUFPR04 = np.sum(previsoesUFPR05xUFPR04 == ocupacoesTesteUFPR04) / len(ocupacoesTesteUFPR04)
acuraciaUFPR05xPUC = np.sum(previsoesUFPR05xPUC == ocupacoesTestePUC) / len(ocupacoesTestePUC)
acuraciaUFPR05xUFPR05 = np.sum(previsoesUFPR05xUFPR05 == ocupacoesTesteUFPR05) / len(ocupacoesTesteUFPR05)

print("Acurácia (treino) UFPR05 x (teste) UFPR04: ", acuraciaUFPR05xUFPR04)
print("Acurácia (treino) UFPR05 x (teste) PUC: ", acuraciaUFPR05xPUC)
print("Acurácia (treino) UFPR05 x (teste) UFPR05: ", acuraciaUFPR05xUFPR05)

# Calculando a acurácia média
acuraciaMediaUFPR04 = (acuraciaUFPR04xUFPR04 + acuraciaUFPR04xPUC + acuraciaUFPR04xUFPR05) / 3
acuraciaMediaPUC = (acuraciaPUCxUFPR04 + acuraciaPUCxPUC + acuraciaPUCxUFPR05) / 3
acuraciaMediaUFPR05 = (acuraciaUFPR05xUFPR04 + acuraciaUFPR05xPUC + acuraciaUFPR05xUFPR05) / 3

print("Acurácia média (treino) x (teste) UFPR04: ", acuraciaMediaUFPR04)
print("Acurácia média (treino) x (teste) PUC: ", acuraciaMediaPUC)
print("Acurácia média (treino) x (teste) UFPR05: ", acuraciaMediaUFPR05)

# Calculando a acurácia geral
acuraciaGeral = (acuraciaMediaUFPR04 + acuraciaMediaPUC + acuraciaMediaUFPR05) / 3

print("Acurácia geral: ", acuraciaGeral)