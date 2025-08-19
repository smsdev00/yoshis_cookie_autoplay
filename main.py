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

MIN_DISTANCE = 80

TOLERANCE = 3

TEMPLATES = {
            'heart' : cv2.imread('./imgs/hearth.png'),
            'yellow' : cv2.imread('./imgs/yellow.png'),
            'green' : cv2.imread('./imgs/green.png'),
        }

# Validar que las imágenes se cargaron correctamente
for name, img in TEMPLATES.items():
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen de la plantilla: ./imgs/{name}.png")
    
# NUEVA FUNCION DE CONVERSION DE IMAGEN DE TABLERO A ARRAY
def screenshot_to_board():
    """
    Convierte una captura de pantalla del tablero en un array NxN dinámico, excluyendo filas no contiguas.
    Args:
        GAME_BOARD_REGION: Tupla (x, y, width, height) con la región aproximada.
        templates: Dicc {tipo: img_template}.
        MIN_DISTANCE : Distancia mínima en píxeles entre cookies contiguas.
        tolerance: Máximo de cookies missing para considerar válido el grid.
    Returns:
        Lista NxN con tipos de cookies del bloque principal, o None si falla.
    """
    cookie_types = list(TEMPLATES.keys())

    # Tomar captura
    try:
        #screenshot = pyautogui.screenshot(region=GAME_BOARD_REGION)
        screenshot = cv2.imread('imgs/static_image.jpg')  # Ej. 'test_board.png'
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error al tomar captura: {e}")
        return None

    # Lista para posiciones y tipos de todas las cookies detectadas
    positions = []
    detected_types = []

    # Detectar múltiples matches para cada template
    for cookie_type, template in TEMPLATES.items():
        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            center_x = pt[0] + template.shape[1] // 2
            center_y = pt[1] + template.shape[0] // 2
            # Evitar duplicados
            if all(np.sqrt((center_x - px)**2 + (center_y - py)**2) > MIN_DISTANCE / 2 for px, py in positions):
                positions.append([center_x, center_y])
                detected_types.append(cookie_type)

    if len(positions) < 9:  # Mínimo para un grid pequeño como 3x3
        print("No se detectaron suficientes cookies.")
        return None

    # Convertir a NumPy
    positions = np.array(positions)

    # Calcular matriz de distancias (sin SciPy)
    diff = positions[:, np.newaxis] - positions[np.newaxis, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    adj_matrix = dists < MIN_DISTANCE * 1.5

    # Connected components con DFS
    visited = np.zeros(len(positions), dtype=bool)
    clusters = []
    for i in range(len(positions)):
        if not visited[i]:
            cluster = []
            stack = [i]
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    cluster.append(node)
                    neighbors = np.where(adj_matrix[node])[0]
                    stack.extend(n for n in neighbors if not visited[n])
            clusters.append(cluster)

    # Cluster más grande
    main_cluster_idx = max(clusters, key=len)
    main_size = len(main_cluster_idx)
    if main_size < 9:
        print("No se encontró bloque principal.")
        return None

    # Calcular N dinámicamente
    N = int(np.round(np.sqrt(main_size)))
    if abs(main_size - N**2) > TOLERANCE:
        print(f"Tamaño no cuadrado válido: {main_size} cookies, esperado cerca de {N**2}.")
        return None

    # Extraer posiciones y tipos del main cluster
    main_positions = positions[main_cluster_idx]
    main_types = [detected_types[i] for i in main_cluster_idx]

    # Ordenar por y, luego x
    sort_idx = np.lexsort((main_positions[:,0], main_positions[:,1]))
    main_positions = main_positions[sort_idx]
    main_types = [main_types[i] for i in sort_idx]

    # Construir board NxN (rellenar None si missing, pero con tolerancia debería estar ok)
    board = []
    for i in range(N):
        row_start = i * N
        row = main_types[row_start:row_start + N]
        if len(row) < N:
            row += [None] * (N - len(row))  # Rellenar si faltan
        board.append(row)

    return board

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

    print("Iniciando screenshot_to_board()")
    screenshot_to_board()
    

if __name__ == "__main__":
    main()