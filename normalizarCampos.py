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
    caminhoArquivoCSV_teste = os.path.join(caminhoUniversidades, 'caracteristicas_teste.csv')
    caminhoArquivoCSV_teste_normalizado = os.path.join(caminhoUniversidades, 'caracteristicas_teste_normalizado.csv')

    # Abrindo arquivo .csv de treino
    with open(caminhoArquivoCSV_treino, 'r') as arquivoCSV_treino:
        leitor = csv.reader(arquivoCSV_treino, delimiter=';')

        # Inicializando vetores
        valoresMinTreino = []
        valoresMaxTreino = []
        
        # Iterando em cada linha do arquivo .csv de treino
        for linha in leitor:
            campos = linha[0].split(';')  # Separar os campos usando ";"

            # Iterando em cada campo e colocando num vetor
            for i, campo_str in enumerate(campos):
                campo = int(campo_str)
                
                if len(valoresMinTreino) <= i:  # Se o vetor ainda não foi inicializado
                    valoresMinTreino.append(campo)
                    valoresMaxTreino.append(campo)
                else:
                    valoresMinTreino[i] = min(valoresMinTreino[i], campo)
                    valoresMaxTreino[i] = max(valoresMaxTreino[i], campo)
    
    # Abrindo arquivo .csv de teste
    with open(caminhoArquivoCSV_teste, 'r') as arquivoCSV_teste:
        leitor = csv.reader(arquivoCSV_teste, delimiter=';')

        # Inicializando vetores
        valoresMinTeste = []
        valoresMaxTeste = []
        
        # Iterando em cada linha do arquivo .csv de treino
        for linha in leitor:
            campos = linha[0].split(';')  # Separar os campos usando ";"

            # Iterando em cada campo e colocando num vetor
            for i, campo_str in enumerate(campos):
                campo = int(campo_str)
                
                if len(valoresMinTeste) <= i:  # Se o vetor ainda não foi inicializado
                    valoresMinTeste.append(campo)
                    valoresMaxTeste.append(campo)
                else:
                    valoresMinTeste[i] = min(valoresMinTeste[i], campo)
                    valoresMaxTeste[i] = max(valoresMaxTeste[i], campo)

    # Zerando ponteiro do arquivo .csv de treino
    with open(caminhoArquivoCSV_treino, 'r') as arquivoCSV_treino:
        leitor = csv.reader(arquivoCSV_treino, delimiter=';')

        # Abrindo arquivo .csv de treino normalizado
        with open(caminhoArquivoCSV_treino_normalizado, 'w') as arquivoCSV_treino_normalizado:
            escritor = csv.writer(arquivoCSV_treino_normalizado, delimiter=';')

            # Iterando nas linhas e normalizando os campos
            for linha in leitor:
                campos = linha[0].split(';')

                camposNormalizados = []  # Inicializando campos normalizados

                # Iterando em cada campo e normalizando
                for i, campo_str in enumerate(campos[:-1]):
                    campo = int(campo_str)

                    # Normalizando o campo
                    campo_normalizado = (campo - valoresMinTreino[i]) / (valoresMaxTreino[i] - valoresMinTreino[i])

                    # Adicionando o campo normalizado à lista de campos normalizados
                    camposNormalizados.append(str(campo_normalizado))

                # Adicionando a ocupação à lista de campos normalizados
                camposNormalizados.append(campos[-1])

                # Criando a linha normalizada como uma string
                linhaNormalizada = ";".join(camposNormalizados)

                # Escrevendo a linha normalizada no arquivo CSV
                escritor.writerow([linhaNormalizada])

    # Zerando ponteiro do arquivo .csv de teste
    with open(caminhoArquivoCSV_teste, 'r') as arquivoCSV_teste:
        leitor = csv.reader(arquivoCSV_teste, delimiter=';')

        # Abrindo arquivo .csv de teste normalizado
        with open(caminhoArquivoCSV_teste_normalizado, 'w') as arquivoCSV_teste_normalizado:
            escritor = csv.writer(arquivoCSV_teste_normalizado, delimiter=';')

            # Iterando nas linhas e normalizando os campos
            for linha in leitor:
                campos = linha[0].split(';')

                camposNormalizados = [] # Inicializando campos normalizados

                # Iterando em cada campo e normalizando
                for i, campo_str in enumerate(campos[:-1]):
                    campo = int(campo_str)

                    # Normalizando o campo
                    campo_normalizado = (campo - valoresMinTeste[i]) / (valoresMaxTeste[i] - valoresMinTeste[i])

                    # Adicionando o campo normalizado à lista de campos normalizados
                    camposNormalizados.append(str(campo_normalizado))
                
                # Adicionando a ocupação à lista de campos normalizados
                camposNormalizados.append(campos[-1])

                # Criando a linha normalizada como uma string
                linhaNormalizada = ";".join(camposNormalizados)

                # Escrevendo a linha normalizada no arquivo CSV
                escritor.writerow([linhaNormalizada])
