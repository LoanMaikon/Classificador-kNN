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
        lbp = feature.local_binary_pattern(imagem, self.numeroPontos, self.raio, method="uniform")
        
        # Calculando o histograma a partir do LBP
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 257), range=(0, 255))

        # Retornando o histograma
        return hist

caminhoDiretorio = '/home/luan/Desktop/PKLot/PKLotSegmented'
caminhoArquivoCSV = '/home/luan/Desktop/PKLot/caracteristicas.csv'

# Abrindo o arquivo .csv
with open(caminhoArquivoCSV, mode="a", newline="") as arquivo:
    escritor = csv.writer(arquivo, delimiter=";")

    for universidades in os.listdir(caminhoDiretorio):
        # Caminho para as universidades 
        caminhoUniversidades = os.path.join(caminhoDiretorio, universidades)

        for climas in os.listdir(caminhoUniversidades):
            # Caminho para os climas
            caminhoClimas = os.path.join(caminhoUniversidades, climas)

            for datas in os.listdir(caminhoClimas):
                # Caminho para as datas
                caminhoDatas = os.path.join(caminhoClimas, datas)

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
                        descritor = LBP(256, 1)
#############################################   256 ou 8????   #########################################

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