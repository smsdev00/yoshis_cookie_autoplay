import pyautogui
import cv2
import numpy as np

templates = {
            'heart' : cv2.imread('./imgs/hearth.png'),
            'yellow' : cv2.imread('./imgs/yellow.png'),
            'green' : cv2.imread('./imgs/green.png'),
        }

def screenshot_to_board(tablero_region, templates, min_distance=20, tolerance=3):
    """
    Convierte una captura de pantalla del tablero en un array NxN dinámico, excluyendo filas no contiguas.
    Args:
        tablero_region: Tupla (x, y, width, height) con la región aproximada.
        templates: Dicc {tipo: img_template}.
        min_distance: Distancia mínima en píxeles entre cookies contiguas.
        tolerance: Máximo de cookies missing para considerar válido el grid.
    Returns:
        Lista NxN con tipos de cookies del bloque principal, o None si falla.
    """
    cookie_types = list(templates.keys())

    # Tomar captura
    try:
        #screenshot = pyautogui.screenshot(region=tablero_region)
        screenshot = cv2.imread('imgs/static_image.jpg')  # Ej. 'test_board.png'
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error al tomar captura: {e}")
        return None

    # Lista para posiciones y tipos de todas las cookies detectadas
    positions = []
    detected_types = []

    # Detectar múltiples matches para cada template
    for cookie_type, template in templates.items():
        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            center_x = pt[0] + template.shape[1] // 2
            center_y = pt[1] + template.shape[0] // 2
            # Evitar duplicados
            if all(np.sqrt((center_x - px)**2 + (center_y - py)**2) > min_distance / 2 for px, py in positions):
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
    adj_matrix = dists < min_distance * 1.5

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
    if abs(main_size - N**2) > tolerance:
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