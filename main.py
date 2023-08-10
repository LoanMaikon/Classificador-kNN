import cv2
import xml.etree.ElementTree as ET
import os
import glob

# Caminho para o diretório principal
caminhoDiretorio= '/home/luan/Desktop/PKLot/PKLot'

for universidades in os.listdir(caminhoDiretorio):
    caminhoUniversidades = os.path.join(caminhoDiretorio, universidades)

    for climas in os.listdir(caminhoUniversidades):
        caminhoClimas = os.path.join(caminhoUniversidades, climas)

        for datas in os.listdir(caminhoClimas):
            caminhoDatas = os.path.join(caminhoClimas, datas)

            # Criando listas para armazenar os arquivos .xml e .jpg
            listaXML = [arq for arq in os.listdir(caminhoDatas) if arq.endswith('.xml')]
            listaJPG = [arq for arq in os.listdir(caminhoDatas) if arq.endswith('.jpg')]

            # Percorrendo as listas
            for i in range(len(listaXML)):
                caminhoXML = os.path.join(caminhoDatas, listaXML[i])
                caminhoJPG = os.path.join(caminhoDatas, listaJPG[i])

                # Carregando o arquivo XML
                xml = ET.parse(caminhoXML).getroot()

                # Carregando a imagem
                imagem = cv2.imread(caminhoJPG)

                # Recortando imagens
                for vaga in xml.findall('.//space'):
                    # ID da vaga e ocupação
                    vagaID = vaga.get('id')
                    ocupacao = vaga.get('occupied')

                    # Coordenadas da vaga e ângulo
                    x = int(vaga.find('rotatedRect').find('center').get('x'))
                    y = int(vaga.find('rotatedRect').find('center').get('y'))
                    w = int(vaga.find('rotatedRect').find('size').get('w'))
                    h = int(vaga.find('rotatedRect').find('size').get('h'))
                    angulo = int(vaga.find('rotatedRect').find('angle').get("d"))

                    # Coordenadas para corte
                    x1 = max(0, x - w // 2)
                    x2 = min(imagem.shape[1], x + w // 2)
                    y1 = max(0, y - h // 2)
                    y2 = min(imagem.shape[0], y + h // 2)

                    # Cortar a imagem
                    recorte = imagem[y1:y2, x1:x2]

                    # Rotação da imagem
                    if angulo >= 45:
                        imagemFinal = cv2.rotate(recorte, cv2.ROTATE_90_CLOCKWISE)
                    else:
                        imagemFinal = recorte

                    # Mostrando a imagem rotacionada
                    cv2.imshow('Imagem Rotacionada', imagemFinal)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()