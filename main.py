import cv2
import easyocr
import numpy as np
import yolov5


def encontrar_cordenadas(url):
    # load model
    model = yolov5.load('keremberke/yolov5m-license-plate')

    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    # set image
    img = url

    # perform inference
    results = model(img, size=640)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # Extraer las coordenadas del cuadro delimitador de la placa del vehículo
    plate_bbox = boxes[0].cpu().numpy()  # Suponiendo que solo hay una detección en la imagen

    # Las coordenadas del cuadro delimitador están en formato (x1, y1, x2, y2)
    x1, y1, x2, y2 = plate_bbox

    # Cargar la imagen usando OpenCV para mostrarla recortada a las coordenadas
    image = cv2.imread(url)
    # Recortar la región de interés (ROI) de la imagen original
    plate_image = image[int(y1):int(y2), int(x1):int(x2)]

    # Mostrar la imagen recortada
    cv2.imshow('Placa Recortada', plate_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Imprimir las coordenadas del cuadro delimitador
    print("Coordenadas del cuadro delimitador de la placa del vehículo:")
    print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

    return plate_image


def read_text_in_plate(image):
    # Verifica si la imagen se ha cargado correctamente
    if image is None:
        return "Error: No se pudo cargar la imagen."

    # Inicializa EasyOCR
    reader = easyocr.Reader(['es'])

    # Reconoce el texto en la región de interés (ROI)
    result = reader.readtext(image)

    # Filtra y extrae el texto reconocido
    recognized_text = ' '.join([text[1] for text in result])  # Extrae el segundo elemento de cada tupla

    return recognized_text


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    url = r'C:\Users\Juan David\PycharmProjects\agentes_inteligentes\carros\carro20.png'
    imagenR = encontrar_cordenadas(url)
    # Lee el texto dentro del cuadro delimitador de la placa
    plate_text = read_text_in_plate(imagenR)

    # Imprime el texto reconocido dentro del cuadro delimitador de la placa
    print("Texto dentro del cuadro delimitador de la placa del vehículo:", plate_text)

