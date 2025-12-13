
import cv2
import numpy as np

def nothing(x):
    pass

def run_color_detector():
    # Intenta cargar una imagen de prueba
    image_path = 'imgs/static_image.jpg'
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: No se pudo cargar la imagen de prueba en '{image_path}'.")
            print("Crea una imagen de captura de pantalla del juego y guárdala como 'imgs/static_image.jpg'.")
            return
        img = cv2.resize(img, (400, 500)) # Redimensionar para que quepa en pantalla
    except Exception as e:
        print(f"Error al cargar o redimensionar la imagen: {e}")
        return

    # Crear una ventana para los trackbars
    cv2.namedWindow('Color Detector')
    cv2.createTrackbar('H Min', 'Color Detector', 0, 179, nothing)
    cv2.createTrackbar('H Max', 'Color Detector', 179, 179, nothing)
    cv2.createTrackbar('S Min', 'Color Detector', 0, 255, nothing)
    cv2.createTrackbar('S Max', 'Color Detector', 255, 255, nothing)
    cv2.createTrackbar('V Min', 'Color Detector', 0, 255, nothing)
    cv2.createTrackbar('V Max', 'Color Detector', 255, 255, nothing)

    print("\n--- Herramienta de Calibración de Colores HSV ---")
    print("Ajusta los deslizadores para aislar un color de cookie.")
    print("La ventana 'Mask' debe mostrar solo las galletas del color deseado en blanco.")
    print("Una vez que estés satisfecho, anota los valores 'min' y 'max'.")
    print("Presiona 'q' para salir y probar con otro color.")
    print("--------------------------------------------------\n")

    while True:
        # Clonar la imagen para no dibujarle encima
        frame = img.copy()
        
        # Convertir a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Obtener valores de los trackbars
        h_min = cv2.getTrackbarPos('H Min', 'Color Detector')
        h_max = cv2.getTrackbarPos('H Max', 'Color Detector')
        s_min = cv2.getTrackbarPos('S Min', 'Color Detector')
        s_max = cv2.getTrackbarPos('S Max', 'Color Detector')
        v_min = cv2.getTrackbarPos('V Min', 'Color Detector')
        v_max = cv2.getTrackbarPos('V Max', 'Color Detector')

        # Crear la máscara
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)

        # Mostrar la máscara y la imagen original
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
    # Imprimir los valores para copiar y pegar en config.py
    print("\nCalibración finalizada.")
    print("Copia estas líneas en tu diccionario 'cookies_colors' en config.py:")
    print(f'"NuevoColor": {{ "min": [{h_min}, {s_min}, {v_min}], "max": [{h_max}, {s_max}, {v_max}] }},')

if __name__ == '__main__':
    run_color_detector()
