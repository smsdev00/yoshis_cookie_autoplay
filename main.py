import cv2
import numpy as np
import pyautogui
import time

# --- CONFIGURACIÓN ---
# El título de la ventana del emulador. Búscalo y ponlo exacto.
EMULATOR_WINDOW_TITLE = "Snes9x 1.60" # Ejemplo, podría ser "ZSNES", "RetroArch", etc.

# Coordenadas del tablero [x, y, ancho, alto].
# ¡ESTO ES LO MÁS IMPORTANTE! Necesitarás encontrar estos valores manualmente.
GAME_BOARD_REGION = (690, 348, 1269, 919) # (x_inicial, y_inicial, ancho, alto) -> ¡REEMPLAZAR!

# --- FUNCIONES PRINCIPALES ---

def capture_game_board(region):
    """
    Toma una captura de pantalla de la región especificada del tablero.
    """
    print("Capturando tablero...")
    try:
        screenshot = pyautogui.screenshot(region=region)
        # Convertir la imagen de Pillow/PyAutoGUI a un formato que OpenCV pueda usar.
        image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        return image
    except Exception as e:
        print(f"Error al capturar la pantalla: {e}")
        return None

def analyze_board(board_image):
    """
    Analiza la imagen del tablero y la convierte en un array 2D
    utilizando template matching.
    """
    print("Analizando tablero...")

    # --- 1. Cargar plantillas y definir IDs ---
    # Asegúrate de que los nombres de archivo coincidan con los que guardaste.
    try:
        templates = {
            1: cv2.imread('./imgs/hearth.png', cv2.IMREAD_UNCHANGED),
            2: cv2.imread('./imgs/green.png', cv2.IMREAD_UNCHANGED),
            3: cv2.imread('./imgs/yellow.png', cv2.IMREAD_UNCHANGED),
        }
    except Exception as e:
        print(f"Error al cargar las imágenes de las plantillas: {e}")
        print("Asegúrate de haber creado y guardado los archivos .png de las cookies.")
        return []

    # Verificar si alguna plantilla no se cargó
    if any(t is None for t in templates.values()):
        print("Error: Una o más plantillas no se pudieron cargar. Revisa los nombres de archivo.")
        return []


    # --- 2. Definir la estructura del tablero ---
    # Yoshi's Cookie para SNES tiene un tablero de 5x5 visible.
    NUM_ROWS = 5
    NUM_COLS = 5
    
    # Obtener las dimensiones de la imagen del tablero capturada
    board_height, board_width, _ = board_image.shape
    
    # Calcular el tamaño de cada celda
    cell_height = board_height // NUM_ROWS
    cell_width = board_width // NUM_COLS

    board_array = []
    match_threshold = 0.8 # Umbral de confianza para la coincidencia (ajustar si es necesario)

    # --- 3. Iterar sobre cada celda del tablero ---
    for r in range(NUM_ROWS):
        row_array = []
        for c in range(NUM_COLS):
            # Coordenadas de la celda actual
            y1 = r * cell_height
            y2 = (r + 1) * cell_height
            x1 = c * cell_width
            x2 = (c + 1) * cell_width

            # Recortar la celda de la imagen del tablero
            cell_image = board_image[y1:y2, x1:x2]

            best_match_cookie_id = 0 # 0 para "desconocido" o "vacío"
            best_match_score = match_threshold

            # --- 4. Comparar la celda con cada plantilla ---
            for cookie_id, template_img in templates.items():
                # Redimensionar la plantilla para que coincida con el tamaño de la celda
                # Esto hace que el matching sea más robusto si hay pequeñas variaciones de tamaño
                resized_template = cv2.resize(template_img, (cell_width, cell_height))
                
                # Realizar el template matching
                result = cv2.matchTemplate(cell_image, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val > best_match_score:
                    best_match_score = max_val
                    best_match_cookie_id = cookie_id
            
            row_array.append(best_match_cookie_id)
        
        board_array.append(row_array)

    # Imprimir el tablero resultante para depuración
    print("Tablero detectado:")
    for row in board_array:
        print(row)

    return board_array
    return board_array

def decide_next_move(board_array):
    """
    Recibe el array del tablero y decide cuál es el mejor movimiento.
    """
    print("Decidiendo movimiento...")
    if not board_array:
        print("El tablero está vacío, no se puede decidir.")
        return None

    # --- LÓGICA DE JUEGO AQUÍ ---
    # Por ahora, no hacemos nada.
    # El objetivo es que devuelva una acción, por ejemplo:
    # {'action': 'move_row', 'index': 3, 'direction': 'right'}
    # {'action': 'move_col', 'index': 1, 'direction': 'up'}
    move = None
    
    return move

def execute_move(move, window):
    """
    Ejecuta el movimiento decidido usando PyAutoGUI para presionar teclas.
    """
    if not move:
        return

    print(f"Ejecutando movimiento: {move}")
    
    # Asegurarse de que la ventana del emulador está activa
    try:
        window.activate()
        time.sleep(0.1) # Pequeña pausa para asegurar que la ventana esté en foco
    except Exception as e:
        print(f"No se pudo activar la ventana del emulador: {e}")
        return

    # --- LÓGICA DE CONTROL AQUÍ ---
    # Aquí traducirías el objeto `move` a pulsaciones de teclas.
    # Ejemplo:
    # if move['action'] == 'move_row':
    #     # Mover cursor a la fila `move['index']`
    #     pyautogui.press('down', presses=move['index']) 
    #     # Mover la fila
    #     pyautogui.press('a') # Suponiendo que 'a' es el botón de acción
    #     pyautogui.press(move['direction'])
    #     pyautogui.press('a') 
    pass

def find_game_window(title):
    """
    Encuentra la ventana del emulador por su título.
    """
    try:
        window = pyautogui.getWindowsWithTitle(title)[0]
        return window
    except IndexError:
        print(f"Error: No se encontró ninguna ventana con el título '{title}'.")
        print("Asegúrate de que el emulador esté abierto y el título sea correcto.")
        return None

# --- BUCLE PRINCIPAL ---

def main():
    """
    El bucle principal que orquesta todo el bot.
    """
    print("Iniciando bot para Yoshi's Cookie...")
    
    game_window = find_game_window(EMULATOR_WINDOW_TITLE)
    if not game_window:
        return # Termina el script si no se encuentra la ventana

    # AHORA que tenemos la ventana, podemos obtener las coordenadas del tablero RELATIVAS a ella.
    # Esto es más robusto que usar coordenadas absolutas de la pantalla.
    global GAME_BOARD_REGION
    GAME_BOARD_REGION = (
        game_window.left, 
        game_window.top, 
        game_window.width, 
        game_window.height
    )
    # **NOTA**: Probablemente necesites ajustar estas coordenadas para que apunten
    # solo al tablero de juego, no a toda la ventana del emulador.

    print(f"Ventana encontrada: {game_window.title}")
    print(f"Usando región de pantalla: {GAME_BOARD_REGION}")
    print("Presiona Ctrl+C en esta terminal para detener el bot.")

    try:
        while True:
            # 1. Capturar el estado del juego
            board_image = capture_game_board(GAME_BOARD_REGION)
            
            if board_image is not None:
                # 2. Analizar la imagen para entender el tablero
                board_state = analyze_board(board_image)

                # 3. Decidir el siguiente movimiento basado en el estado
                chosen_move = decide_next_move(board_state)

                # 4. Ejecutar el movimiento
                if chosen_move:
                    execute_move(chosen_move, game_window)
            
            # Esperar un poco antes de la siguiente iteración para no sobrecargar la CPU
            time.sleep(1) 

    except KeyboardInterrupt:
        print("\nBot detenido por el usuario.")
    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}")

if __name__ == "__main__":
    main()