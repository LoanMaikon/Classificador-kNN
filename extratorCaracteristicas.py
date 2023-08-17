import cv2
import numpy as np
import os
import csv
from skimage import feature

class LBP:
    # Construtor da classe
    def __init__(self, numeroPontos, raio):
        self.numeroPontos = numeroPontos
        self.raio = raio

    # Construindo o histograma
    def construirHistograma(self, imagem, eps=1e-7):
        # Calculando o LBP da imagem
        lbp = feature.local_binary_pattern(imagem, self.numeroPontos, self.raio, method="nri_uniform")
        
        # Calculando o histograma a partir do LBP (o método nri_uniform retorna 59 valores possíveis)
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 60), range=(0, 255))

        # Retornando o histograma
        return hist

caminhoDiretorio = '/home/luan/Desktop/PKLot/PKLotSegmented'

for universidades in os.listdir(caminhoDiretorio):
    # Caminho para as universidades 
    caminhoUniversidades = os.path.join(caminhoDiretorio, universidades)

    caminhoArquivoCSV_treino = os.path.join(caminhoUniversidades, 'caracteristicas_treino.csv')
    caminhoArquivoCSV_teste = os.path.join(caminhoUniversidades, 'caracteristicas_teste.csv')

    for climas in os.listdir(caminhoUniversidades):
        # Caminho para os climas
        caminhoClimas = os.path.join(caminhoUniversidades, climas)

        i = 0 # Para iterar entre os arquivos de treino e teste

        for datas in os.listdir(caminhoClimas):
            # Caminho para as datas
            caminhoDatas = os.path.join(caminhoClimas, datas)

            if i % 2 == 0:
                caminhoArquivoCSV = caminhoArquivoCSV_treino # Caso i seja par, usa data para treino
            else:
                caminhoArquivoCSV = caminhoArquivoCSV_teste # Caso i seja ímpar, usa data para teste

            with open(caminhoArquivoCSV, 'a') as arquivoCSV:
                escritor = csv.writer(arquivoCSV, delimiter=';')

                for ocupacao in os.listdir(caminhoDatas):
                    # Caminho para as ocupações
                    caminhoOcupacoes = os.path.join(caminhoDatas, ocupacao)

                    for imagens in os.listdir(caminhoOcupacoes):
                        # Caminho para as imagens
                        caminhoImagens = os.path.join(caminhoOcupacoes, imagens)

                        # Carregando a imagem e transformando em escala de cinza
                        imagem = cv2.imread(caminhoImagens)
                        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

                        # Criando descritor LBP (recebendo informações)
                        descritor = LBP(8, 1)

                        # Criando vetor de características
                        vetorCaracteristicas = descritor.construirHistograma(imagem)

                        # Criando a string de escrita do arquivo .csv
                        escrita = ";".join(map(str, vetorCaracteristicas))
                        if ocupacao == "Occupied":
                            escrita += ";1"
                        else:
                            escrita += ";0"

                        # Escrevendo a string no arquivo .csv
                        escritor.writerow([escrita])


            i += 1 #Acrescenta ao iterador