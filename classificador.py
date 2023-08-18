import os
import csv
import numpy as np

# Função para calcular a distância euclidiana entre dois vetores e retornar o vetor resultante
def distanciaEuclidiana(vetor1, vetor2):
    return np.sqrt(np.sum((vetor1 - vetor2)**2))

# Função de classificação K-NN
def kNN(dadosTreino, ocupacoes, caracteristicasImagemTeste, kValor):
    # Vetor de distâncias
    distancias = []

    for i in range(len(dadosTreino)):
        dist = distanciaEuclidiana(dadosTreino[i], caracteristicasImagemTeste)
        distancias.append((dist, ocupacoes[i]))

    distancias.sort(key=lambda x: x[0])  # Ordenar com base nas distâncias
    k_vizinhos = distancias[:kValor]  # Selecionar os k vizinhos mais próximos

    # Contagem das classes dos vizinhos mais próximos
    contagem_classes = {0: 0, 1: 0}  # Supondo que 0 é vaga vazia e 1 é vaga ocupada
    for dist, label in k_vizinhos:
        contagem_classes[label] += 1

    # Escolha da classe mais comum entre os vizinhos
    resultado = max(contagem_classes, key=contagem_classes.get)
    return resultado

# Definindo K-NN de 3 vizinhos
kValor = 3

# Vetores para armazenar os dados de treinamento
dadosTreinoPUC = []
dadosTreinoUFPR04 = []
dadosTreinoUFPR05 = []
ocupacoesPUC = []
ocupacoesUFPR04 = []
ocupacoesUFPR05 = []

# Caminho para o diretório das imagens
caminhoDiretorio = '/home/luan/Desktop/PKLot/PKLotSegmented'

# Retirando dados da PUC
caminhoPUC = os.path.join(caminhoDiretorio, 'PUC')
caminhoArquivoCSV_treino_normalizado_PUC = os.path.join(caminhoPUC, 'caracteristicas_treino_normalizado.csv')

with open(caminhoArquivoCSV_treino_normalizado_PUC, 'r') as arquivoCSV_treino_normalizado_PUC:
    leitor = csv.reader(arquivoCSV_treino_normalizado_PUC, delimiter=';')  # Definir o separador como ";"
    
    # Iterando nas linhas do arquivo .csv para retirar características e ocupação
    for linha in leitor:
        # Campos já é uma lista de campos separados por ";"
        campos = linha

        # Colocando as características da linha em um vetor
        caracteristicas = np.array([float(campo) for campo in campos[:-1]])

        # Retirando a ocupação
        ocupacao = campos[-1]

        # Adicionando os dados de treino e a ocupação nas listas
        dadosTreinoPUC.append(caracteristicas)
        ocupacoesPUC.append(ocupacao)



# Retirando dados da UFPR04
caminhoUFPR04 = os.path.join(caminhoDiretorio, 'UFPR04')
caminhoArquivoCSV_treino_normalizado_UFPR04 = os.path.join(caminhoUFPR04, 'caracteristicas_treino_normalizado.csv')

with open(caminhoArquivoCSV_treino_normalizado_UFPR04, 'r') as arquivoCSV_treino_normalizado_UFPR04:
    leitor = csv.reader(arquivoCSV_treino_normalizado_UFPR04)
    
    # Iterando nas linhas do arquivo .csv para retirar características e ocupação
    for linha in leitor:
        # Dividir a linha em campos usando o separador ";"
        campos = linha

        # Colocando as características da linha em um vetor
        caracteristicas = np.array([float(campo) for campo in campos[:-1]])

        # Retirando a ocupação
        ocupacao = campos[-1]

        # Adicionando os dados de treino e a ocupação nas listas
        dadosTreinoUFPR04.append(caracteristicas)
        ocupacoesUFPR04.append(ocupacao)
    
# Retirando dados da UFPR05
caminhoUFPR05 = os.path.join(caminhoDiretorio, 'UFPR05')
caminhoArquivoCSV_treino_normalizado_UFPR05 = os.path.join(caminhoUFPR05, 'caracteristicas_treino_normalizado.csv')

with open(caminhoArquivoCSV_treino_normalizado_UFPR05, 'r') as arquivoCSV_treino_normalizado_UFPR05:
    leitor = csv.reader(arquivoCSV_treino_normalizado_UFPR05)
    
    # Iterando nas linhas do arquivo .csv para retirar características e ocupação
    for linha in leitor:
        # Dividir a linha em campos usando o separador ";"
        campos = linha

        # Colocando as características da linha em um vetor
        caracteristicas = np.array([float(campo) for campo in campos[:-1]])

        # Retirando a ocupação
        ocupacao = campos[-1]

        # Adicionando os dados de treino e a ocupação nas listas
        dadosTreinoUFPR05.append(caracteristicas)
        ocupacoesUFPR05.append(ocupacao)

# Convertendo em vetores numpy para cálculos mais eficientes
dadosTreinoPUC = np.array(dadosTreinoPUC)
ocupacoesPUC = np.array(ocupacoesPUC)
dadosTreinoUFPR04 = np.array(dadosTreinoUFPR04)
ocupacoesUFPR04 = np.array(ocupacoesUFPR04)
dadosTreinoUFPR05 = np.array(dadosTreinoUFPR05)
ocupacoesUFPR05 = np.array(ocupacoesUFPR05)

print(dadosTreinoPUC)

# Teste para PUC
caminhoArquivoCSV_teste_PUC = os.path.join(caminhoPUC, 'caracteristicas_teste.csv')

with open(caminhoArquivoCSV_teste_PUC, 'r') as arquivoCSV_teste_PUC:
    leitor = csv.reader(arquivoCSV_teste_PUC)
    
    # Iterando nas linhas do arquivo .csv de teste
    for linha in leitor:
        campos = linha[0].split(';')

        caracteristicasImagemTeste = np.array([int(campo) for campo in campos[:-1]])

        resultado = kNN(dadosTreinoPUC, ocupacoesPUC, caracteristicasImagemTeste, kValor)
        print(resultado)
