import os
import csv
import numpy as np

caminhoDiretorio = '/home/luan/Desktop/PKLot/PKLotSegmented'

for universidades in os.listdir(caminhoDiretorio):
    # Caminho para as universidades 
    caminhoUniversidades = os.path.join(caminhoDiretorio, universidades)

    # Caminho para o arquivo .csv de treino e treino normalizado
    caminhoArquivoCSV_treino = os.path.join(caminhoUniversidades, 'caracteristicas_treino.csv')
    caminhoArquivoCSV_treino_normalizado = os.path.join(caminhoUniversidades, 'caracteristicas_treino_normalizado.csv')

    # Abrindo arquivo .csv de treino
    with open(caminhoArquivoCSV_treino, 'r') as arquivoCSV_treino:
        leitor = csv.reader(arquivoCSV_treino, delimiter=';')

        # Inicializando vetores
        valoresMin = []
        valoresMax = []
        
        # Iterando em cada linha do arquivo .csv de treino
        for linha in leitor:
            campos = linha[0].split(';')  # Separar os campos usando ";"

            # Iterando em cada campo e colocando num vetor
            for i, campo_str in enumerate(campos):
                campo = int(campo_str)
                
                if len(valoresMin) <= i:  # Se o vetor ainda não foi inicializado
                    valoresMin.append(campo)
                    valoresMax.append(campo)
                else:
                    valoresMin[i] = min(valoresMin[i], campo)
                    valoresMax[i] = max(valoresMax[i], campo)

    # Zerando ponteiro do arquivo .csv de treino
    with open(caminhoArquivoCSV_treino, 'r') as arquivoCSV_treino:
        leitor = csv.reader(arquivoCSV_treino, delimiter=';')

        # Abrindo arquivo .csv de treino normalizado
        with open(caminhoArquivoCSV_treino_normalizado, 'w') as arquivoCSV_treino_normalizado:
            escritor = csv.writer(arquivoCSV_treino_normalizado, delimiter=';')

            # Iterando nas linhas e normalizando os campos
            for linha in leitor:
                campos = linha[0].split(';')

                linhaNormalizada = [] # Inicializando linha normalizada

                # Iterando em cada campo e normalizando
                for i, campo_str in enumerate(campos[:-1]):
                    campo = int(campo_str)

                    # Normalizando o campo
                    campoNormalizado = (campo - valoresMin[i]) / (valoresMax[i] - valoresMin[i])

                    # Adicionando o campo normalizado na linha normalizada
                    linhaNormalizada.append(campoNormalizado)

                # Adicionando a ocupação na linha normalizada
                linhaNormalizada.append(campos[-1])

                # Escrevendo a linha normalizada no arquivo CSV
                escritor.writerow(linhaNormalizada)
