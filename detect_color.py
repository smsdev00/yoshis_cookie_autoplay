import cv2
import numpy as np

# Ruta de la imagen, asegúrate de que esté en la misma carpeta que el script
IMAGE_PATH = './imgs/static_image.jpg'

def nothing(x):
    pass

def hsv_color_selector():
    # Cargar la imagen
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("❌ Error: No se pudo cargar la imagen. Verifica la ruta.")
        return

    # Crear una ventana para las barras deslizantes
    cv2.namedWindow("HSV Trackbars", cv2.WINDOW_AUTOSIZE)

    # Crear las barras deslizantes para los rangos de HSV
    cv2.createTrackbar("H_min", "HSV Trackbars", 0, 179, nothing)
    cv2.createTrackbar("S_min", "HSV Trackbars", 0, 255, nothing)
    cv2.createTrackbar("V_min", "HSV Trackbars", 0, 255, nothing)
    cv2.createTrackbar("H_max", "HSV Trackbars", 179, 179, nothing)
    cv2.createTrackbar("S_max", "HSV Trackbars", 255, 255, nothing)
    cv2.createTrackbar("V_max", "HSV Trackbars", 255, 255, nothing)

    # Bucle principal para actualizar la imagen
    while True:
        # Convertir la imagen a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Obtener las posiciones de las barras deslizantes
        h_min = cv2.getTrackbarPos("H_min", "HSV Trackbars")
        s_min = cv2.getTrackbarPos("S_min", "HSV Trackbars")
        v_min = cv2.getTrackbarPos("V_min", "HSV Trackbars")
        h_max = cv2.getTrackbarPos("H_max", "HSV Trackbars")
        s_max = cv2.getTrackbarPos("S_max", "HSV Trackbars")
        v_max = cv2.getTrackbarPos("V_max", "HSV Trackbars")

        # Crear los arrays de NumPy con los rangos HSV
        lower_range = np.array([h_min, s_min, v_min])
        upper_range = np.array([h_max, s_max, v_max])

        # Crear la máscara binaria para el color seleccionado
        mask = cv2.inRange(hsv, lower_range, upper_range)

        # Mostrar la imagen original y la máscara
        cv2.imshow("Original Image", image)
        cv2.imshow("Mask", mask)
        
        # Esperar la pulsación de una tecla para salir
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Presiona la tecla 'Esc' para salir
            break

    # Imprimir los valores finales
    print("---------------------------------------")
    print(f"✅ Rango HSV final:")
    print(f"  lower_range = np.array([{h_min}, {s_min}, {v_min}])")
    print(f"  upper_range = np.array([{h_max}, {s_max}, {v_max}])")
    print("---------------------------------------")
    
    # Destruir todas las ventanas
    cv2.destroyAllWindows()

if __name__ == "__main__":
    hsv_color_selector()