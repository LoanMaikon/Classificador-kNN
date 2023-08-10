import cv2
import xml.etree.ElementTree as ET

# Caminho para a imagem
caminho_imagem = '/home/luan/Desktop/PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_07_44_29.jpg'

# Carregar a imagem
imagem = cv2.imread(caminho_imagem)

# Carregando o arquivo XML
xml = ET.parse('/home/luan/Desktop/PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_07_44_29.xml').getroot()

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
