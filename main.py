import cv2
import xml.etree.ElementTree as ET
import os
import glob

# Caminho para o diretório principal
caminhoDiretorio= '/home/luan/Desktop/PKLot/PKLot'
caminhoNovo = '/home/luan/Desktop/PKLot/PKLotSegmentedNovo'

# Criando diretório
if not os.path.exists(caminhoNovo):
    os.makedirs(caminhoNovo)

for universidades in os.listdir(caminhoDiretorio):
    # Caminho para as universidades 
    caminhoUniversidades = os.path.join(caminhoDiretorio, universidades)
    caminhoUniversidadesNovo = os.path.join(caminhoNovo, universidades)

    #criando diretório
    os.makedirs(caminhoUniversidadesNovo, exist_ok=True)

    for climas in os.listdir(caminhoUniversidades):
        # Caminho para os climas
        caminhoClimas = os.path.join(caminhoUniversidades, climas)
        caminhoClimasNovo = os.path.join(caminhoUniversidadesNovo, climas)

        # Criando diretório
        os.makedirs(caminhoClimasNovo, exist_ok=True)

        for datas in os.listdir(caminhoClimas):
            # Caminho para as datas
            caminhoDatas = os.path.join(caminhoClimas, datas)
            caminhoDatasNovo = os.path.join(caminhoClimasNovo, datas)

            # Criando diretório
            os.makedirs(caminhoDatasNovo, exist_ok=True)

            # Criando listas para armazenar os arquivos .xml e .jpg
            listaXML = [arq for arq in os.listdir(caminhoDatas) if arq.endswith('.xml')]
            listaJPG = [arq for arq in os.listdir(caminhoDatas) if arq.endswith('.jpg')]

            # Criando pastas Empty e Occupied
            caminhoEmpty = os.path.join(caminhoDatasNovo, 'Empty')
            caminhoOccupied = os.path.join(caminhoDatasNovo, 'Occupied')
            os.makedirs(caminhoEmpty, exist_ok=True)
            os.makedirs(caminhoOccupied, exist_ok=True)

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
                    vagaID = vaga.get('id').zfill(3)
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

                    # Criando nome da imagem
                    nomeImagem = datas + "#" + vagaID + ".jpg"

                    # Colocando imagem em Empty ou Occupied
                    if ocupacao == '0':
                        cv2.imwrite(os.path.join(caminhoEmpty, "#" + vagaID + '.jpg'), imagemFinal)
                    elif ocupacao == '1':
                        cv2.imwrite(os.path.join(caminhoOccupied, "#" + vagaID + '.jpg'), imagemFinal)

                    