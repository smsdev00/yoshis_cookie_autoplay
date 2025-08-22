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
    """
    cookies = {
        "Verde": {"cant": 0, "coords": []},
        "Amarillo": {"cant": 0, "coords": []},
        "Rojo": {"cant": 0, "coords": []}
    }

    imagen = cv2.imread(imagen_path)
    if imagen is None:
        print("❌ Error: No se pudo cargar la imagen. Verifica la ruta.")
        return None

    imagen_con_detecciones = imagen.copy()
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    
    colores_dibujo = {
        'Verde': (235, 20, 160),      # Rosa fluo
        'Rojo': (255, 255, 255),      # Blanco
        'Amarillo': (0, 255, 255),    # Amarillo
    }

    cv2.rectangle(
        imagen_con_detecciones,
        (area_juego['x_min'], area_juego['y_min']),
        (area_juego['x_max'], area_juego['y_max']),
        (255, 0, 0),
        2,
    )

    for nombre_color, rangos in rangos_colores.items():
        mascara = cv2.inRange(hsv, np.array(rangos['min']), np.array(rangos['max']))
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if nombre_color == 'Verde':
                perimetro = cv2.arcLength(contorno, True)
                if perimetro > 0:
                    circularidad = 4 * np.pi * area / (perimetro * perimetro)
                    if circularidad < 0.6:
                        continue
                x, y, w, h = cv2.boundingRect(contorno)
                aspect_ratio = float(w) / h
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue
                    
            if area > 600:
                momentos = cv2.moments(contorno)
                if momentos["m00"] != 0:
                    cx = int(momentos["m10"] / momentos["m00"])
                    cy = int(momentos["m01"] / momentos["m00"])
                    
                    if (area_juego['x_min'] <= cx <= area_juego['x_max'] and
                        area_juego['y_min'] <= cy <= area_juego['y_max']):
                        
                        # Almacenar la cookie en el diccionario
                        cookies[nombre_color]["cant"] += 1
                        cookies[nombre_color]["coords"].append((cx, cy))

                        color_punto = colores_dibujo.get(nombre_color, (255, 255, 255))
                        cv2.circle(imagen_con_detecciones, (cx, cy), 5, color_punto, -1)
                        print(f"✅ {nombre_color} cookie detectada en: ({cx}, {cy})")

    # --- IMPRIMIR RESUMEN DE COOKIES ---
    print("\n--- Resumen de Detecciones ---")
    total_cookies = 0
    for nombre, datos in cookies.items():
        print(f"🍪 {nombre}: {datos['cant']} cookies")
        total_cookies += datos['cant']
    
    print(f"\n✅ Total de cookies detectadas: {total_cookies}")
    
    return imagen_con_detecciones

if __name__ == '__main__':
    imagen_final = visualizar_deteccion_cookies('imgs/static_image.jpg', CONF["cookies_colors"], CONF["game_area"])
    
    if imagen_final is not None:
        cv2.imshow("Detección de Cookies", imagen_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()