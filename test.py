from config import CONF
import cv2
import numpy as np

def visualizar_deteccion_cookies(
    imagen_path: str, 
    rangos_colores: dict, 
    area_juego: dict
) -> np.ndarray:
    """
    Toma una imagen, rangos de color y un área de juego para detectar y
    visualizar galletas en la imagen.

    Args:
        imagen_path (str): La ruta del archivo de la imagen de entrada.
        rangos_colores (dict): Un diccionario con los rangos HSV de cada color
                               de cookie.
        area_juego (dict): Un diccionario que define el área de juego.

    Returns:
        np.ndarray: La imagen con las visualizaciones.
    """
    cookies = {
        "Verde":{"cant":0,"coords":[]},
        "Amarillo":{"cant":0,"coords":[]},
        "Rojo":{"cant":0,"coords":[]}
    }
    # Cargar la imagen
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        print("❌ Error: No se pudo cargar la imagen. Verifica la ruta.")
        return None

    # Copia de la imagen para dibujar sobre ella
    imagen_con_detecciones = imagen.copy()

    # Convertir a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir colores de visualización para cada tipo de cookie (en BGR)
    colores_dibujo = {
        'Verde': (235, 20, 160),    # Rosa fluo
        'Rojo': (255, 255, 255),     # Rojo
        'Amarillo': (0, 255, 255), # Amarillo
        # Añade más colores si tienes otros tipos de cookies
    }

    # Dibujar el recuadro del área de juego
    cv2.rectangle(
        imagen_con_detecciones,
        (area_juego['x_min'], area_juego['y_min']),
        (area_juego['x_max'], area_juego['y_max']),
        (255, 0, 0),  # Color del recuadro (azul)
        2,            # Grosor de la línea
    )

    # Iterar sobre cada tipo de cookie para detectarlas
    for nombre_color, rangos in rangos_colores.items():
        # Crear la máscara para el color actual
        mascara = cv2.inRange(hsv, np.array(rangos['min']), np.array(rangos['max']))

        # Encontrar contornos
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Iterar sobre cada contorno para encontrar el centro
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if(nombre_color == 'Verde'):
               
                # --- NUEVOS FILTROS DE FORMA ---
                # Filtro 1: por circularidad (para la forma redonda de las galletas)
                perimetro = cv2.arcLength(contorno, True)
                if perimetro > 0:
                    circularidad = 4 * np.pi * area / (perimetro * perimetro)
                    if circularidad < 0.6: # Valor ajustado, puedes subirlo a 0.7 o 0.8
                        continue

                # Filtro 2: por relación de aspecto (relación ancho/alto)
                x, y, w, h = cv2.boundingRect(contorno)
                aspect_ratio = float(w) / h
                if aspect_ratio < 0.5 or aspect_ratio > 2.0: # Valores amplios para formas casi cuadradas/circulares
                    continue
            if area > 600:  # Filtrar por área para evitar ruido
                momentos = cv2.moments(contorno)
                if momentos["m00"] != 0:
                    cx = int(momentos["m10"] / momentos["m00"])
                    cy = int(momentos["m01"] / momentos["m00"])
                     # Filtrar por el área de juego
                    if (area_juego['x_min'] <= cx <= area_juego['x_max'] and
                        area_juego['y_min'] <= cy <= area_juego['y_max']):
                        
                        # Dibujar un círculo en la posición de la cookie
                        color_punto = colores_dibujo.get(nombre_color, (255, 255, 255)) # Default blanco
                        cv2.circle(imagen_con_detecciones, (cx, cy), 5, color_punto, -1)
                        print(f"✅ {nombre_color} cookie detectada en: ({cx}, {cy})")

    print(cookies)
    return imagen_con_detecciones

if __name__ == '__main__':


    # Ejecutar la función
    imagen_final = visualizar_deteccion_cookies('imgs/static_image.jpg', CONF["cookies_colors"],  CONF["game_area"])
    
    if imagen_final is not None:
        cv2.imshow("Detección de Cookies", imagen_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()