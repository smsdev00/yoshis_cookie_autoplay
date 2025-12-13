from config import CONF
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
import os

@dataclass
class Cookie:
    """Representa una cookie detectada."""
    color: str
    x: int
    y: int
    row: int = -1
    col: int = -1


class ImprovedCookieDetector:
    """Sistema mejorado de detección de cookies con clustering inteligente."""
    
    def __init__(self, config: dict):
        self.config = config
        self.cookies_colors = config["cookies_colors"]
        self.game_area = config["game_area"]
        self.images_path = config["images_path"]
        
        # Parámetros de clustering
        self.CLUSTER_TOLERANCE = 35  # Tolerancia para agrupar cookies en filas/columnas
        self.MIN_SAMPLES = 2  # Mínimo de cookies para formar un cluster
        
        # Colores para visualización
        self.colores_dibujo = {
            'Verde': (0, 255, 0),
            'Rojo': (0, 0, 255),
            'Amarillo': (0, 255, 255),
        }
        
        self.COLOR_MAP = {
            "Verde": 1,
            "Rojo": 2,
            "Amarillo": 3,
        }

    def detectar_cookies(self, imagen_path: str) -> List[Cookie]:
        """Detecta todas las cookies en la imagen."""
        cookies = []
        
        imagen = cv2.imread(imagen_path)
        if imagen is None:
            raise ValueError(f"Error: No se pudo cargar {imagen_path}")

        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        print("\n[INFO] Detectando cookies...")
        for nombre_color, rangos in self.cookies_colors.items():
            contornos_validos = self._filtrar_contornos(hsv, rangos, nombre_color)
            
            for contorno in contornos_validos:
                cx, cy = self._obtener_centro(contorno)
                
                if self._esta_en_area_juego(cx, cy):
                    cookie = Cookie(color=nombre_color, x=cx, y=cy)
                    cookies.append(cookie)
        
        print(f"[INFO] Total cookies detectadas: {len(cookies)}")
        return cookies

    def _filtrar_contornos(self, hsv: np.ndarray, rangos: dict, nombre_color: str) -> List:
        """Filtra contornos por color y forma."""
        mascara = cv2.inRange(hsv, np.array(rangos['min']), np.array(rangos['max']))
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contornos_validos = []
        
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area <= self.config["min_area_cookie"]:
                continue
                
            if nombre_color == 'Verde' and not self._es_forma_circular(contorno):
                continue
                    
            if not self._tiene_aspect_ratio_valido(contorno):
                continue
                
            contornos_validos.append(contorno)
            
        return contornos_validos

    def _es_forma_circular(self, contorno) -> bool:
        """Verifica forma circular."""
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        if perimetro == 0:
            return False
        circularidad = 4 * np.pi * area / (perimetro * perimetro)
        return circularidad >= 0.6

    def _tiene_aspect_ratio_valido(self, contorno) -> bool:
        """Verifica aspect ratio válido."""
        x, y, w, h = cv2.boundingRect(contorno)
        aspect_ratio = float(w) / h
        return 0.5 <= aspect_ratio <= 2.0

    def _obtener_centro(self, contorno) -> Tuple[int, int]:
        """Calcula centro del contorno."""
        momentos = cv2.moments(contorno)
        if momentos["m00"] == 0:
            return (0, 0)
        cx = int(momentos["m10"] / momentos["m00"])
        cy = int(momentos["m01"] / momentos["m00"])
        return cx, cy

    def _esta_en_area_juego(self, x: int, y: int) -> bool:
        """Verifica si está en área de juego."""
        return (self.game_area['x_min'] <= x <= self.game_area['x_max'] and
                self.game_area['y_min'] <= y <= self.game_area['y_max'])

    def _filtrar_cookies_no_jugables(self, cookies: List[Cookie], filas_labels: np.ndarray, 
                                      columnas_labels: np.ndarray) -> List[Cookie]:
        """
        Filtra cookies que no son parte del tablero jugable.
        Criterios:
        1. Excluir filas con menos de 3 cookies (filas incompletas/cayendo)
        2. Excluir cookies que no tienen suficientes vecinas en su columna
        3. Verificar densidad de la fila
        """
        # Agrupar cookies por fila
        filas_dict = {}
        for i, cookie in enumerate(cookies):
            fila_label = filas_labels[i]
            if fila_label != -1:
                if fila_label not in filas_dict:
                    filas_dict[fila_label] = []
                filas_dict[fila_label].append((i, cookie))
        
        # Identificar filas jugables (filas con al menos 3 cookies)
        MIN_COOKIES_POR_FILA = 3
        filas_jugables = set()
        
        for fila_label, cookies_en_fila in filas_dict.items():
            if len(cookies_en_fila) >= MIN_COOKIES_POR_FILA:
                filas_jugables.add(fila_label)
        
        if not filas_jugables:
            return []
        
        # Calcular densidad de cada fila (cookies por unidad de ancho)
        densidades_filas = {}
        for fila_label in filas_jugables:
            cookies_fila = [c for _, c in filas_dict[fila_label]]
            xs = [c.x for c in cookies_fila]
            ancho_fila = max(xs) - min(xs) if len(xs) > 1 else 1
            densidad = len(cookies_fila) / max(ancho_fila, 1)
            densidades_filas[fila_label] = densidad
        
        # Identificar el "core" del tablero (filas con mayor densidad)
        if densidades_filas:
            densidad_media = np.median(list(densidades_filas.values()))
            UMBRAL_DENSIDAD = densidad_media * 0.6  # 60% de la densidad media
            
            # Filtrar filas con baja densidad (probablemente cayendo)
            filas_core = {
                label for label, densidad in densidades_filas.items() 
                if densidad >= UMBRAL_DENSIDAD
            }
        else:
            filas_core = filas_jugables
        
        # ADICIONAL: Excluir la fila superior si hay una brecha grande con la siguiente
        if len(filas_core) > 1:
            # Obtener posiciones Y de cada fila
            filas_y = {}
            for fila_label in filas_core:
                cookies_fila = [c for _, c in filas_dict[fila_label]]
                y_promedio = np.mean([c.y for c in cookies_fila])
                filas_y[fila_label] = y_promedio
            
            # Ordenar filas por Y (arriba a abajo)
            filas_ordenadas = sorted(filas_y.items(), key=lambda x: x[1])
            
            # Verificar si la primera fila está muy separada de las demás
            if len(filas_ordenadas) >= 2:
                gap_primera = filas_ordenadas[1][1] - filas_ordenadas[0][1]
                
                # Calcular gaps promedio entre otras filas
                gaps = []
                for i in range(1, len(filas_ordenadas) - 1):
                    gap = filas_ordenadas[i + 1][1] - filas_ordenadas[i][1]
                    gaps.append(gap)
                
                if gaps:
                    gap_promedio = np.mean(gaps)
                    # Si el gap de la primera fila es > 1.5x el promedio, excluirla
                    if gap_primera > gap_promedio * 1.5:
                        print(f"[INFO] Excluyendo fila superior (gap: {gap_primera:.1f} vs promedio: {gap_promedio:.1f})")
                        filas_core.discard(filas_ordenadas[0][0])
        
        # Filtrar cookies
        cookies_jugables = []
        
        for i, cookie in enumerate(cookies):
            fila_label = filas_labels[i]
            if fila_label in filas_core:
                cookies_jugables.append(cookie)
        
        print(f"[INFO] Cookies jugables: {len(cookies_jugables)}/{len(cookies)}")
        print(f"[INFO] Filas jugables detectadas: {len(filas_core)}")
        
        return cookies_jugables

    def construir_grilla_inteligente(self, cookies: List[Cookie]) -> Tuple[np.ndarray, Dict]:
        """
        Construye la grilla usando clustering para detectar filas y columnas automáticamente.
        """
        if not cookies:
            return np.array([[]]), {}
        
        # Extraer coordenadas
        coords_y = np.array([c.y for c in cookies]).reshape(-1, 1)
        coords_x = np.array([c.x for c in cookies]).reshape(-1, 1)
        
        # Clustering en Y (filas)
        clustering_y = DBSCAN(eps=self.CLUSTER_TOLERANCE, min_samples=self.MIN_SAMPLES)
        filas_labels = clustering_y.fit_predict(coords_y)
        
        # Clustering en X (columnas)
        clustering_x = DBSCAN(eps=self.CLUSTER_TOLERANCE, min_samples=self.MIN_SAMPLES)
        columnas_labels = clustering_x.fit_predict(coords_x)
        
        # NUEVO: Filtrar filas no jugables (filas cayendo o incompletas)
        cookies_jugables = self._filtrar_cookies_no_jugables(cookies, filas_labels, columnas_labels)
        
        if not cookies_jugables:
            print("[WARN] No se encontraron cookies jugables")
            return np.array([[]]), {}
        
        # Recalcular clustering solo con cookies jugables
        coords_y_jugables = np.array([c.y for c in cookies_jugables]).reshape(-1, 1)
        coords_x_jugables = np.array([c.x for c in cookies_jugables]).reshape(-1, 1)
        
        filas_labels = clustering_y.fit_predict(coords_y_jugables)
        columnas_labels = clustering_x.fit_predict(coords_x_jugables)
        
        # Actualizar referencias
        cookies_originales = cookies
        cookies = cookies_jugables
        
        # Verificar que se detectaron clusters válidos
        num_filas = len(set(filas_labels)) - (1 if -1 in filas_labels else 0)
        num_cols = len(set(columnas_labels)) - (1 if -1 in columnas_labels else 0)
        
        if num_filas == 0 or num_cols == 0:
            print("[WARN] No se detectaron suficientes filas/columnas")
            return np.array([[]]), {}
        
        # Calcular centroides de clusters
        filas_centroids = {}
        for label in set(filas_labels):
            if label != -1:
                indices = np.where(filas_labels == label)[0]
                centroid = np.mean(coords_y_jugables[indices])
                filas_centroids[label] = centroid
        
        columnas_centroids = {}
        for label in set(columnas_labels):
            if label != -1:
                indices = np.where(columnas_labels == label)[0]
                centroid = np.mean(coords_x_jugables[indices])
                columnas_centroids[label] = centroid
        
        # Ordenar clusters por posición
        filas_ordenadas = sorted(filas_centroids.items(), key=lambda x: x[1])
        columnas_ordenadas = sorted(columnas_centroids.items(), key=lambda x: x[1])
        
        # Crear mapeo de labels a índices de grilla
        fila_label_to_idx = {label: idx for idx, (label, _) in enumerate(filas_ordenadas)}
        col_label_to_idx = {label: idx for idx, (label, _) in enumerate(columnas_ordenadas)}
        
        # Asignar posiciones de grilla a cada cookie
        cookies_excluidas = []
        for i, cookie in enumerate(cookies):
            fila_label = filas_labels[i]
            col_label = columnas_labels[i]
            
            if fila_label != -1 and col_label != -1:
                cookie.row = fila_label_to_idx[fila_label]
                cookie.col = col_label_to_idx[col_label]
            else:
                cookies_excluidas.append(cookie)
        
        # Agregar cookies no jugables a la lista de excluidas
        cookies_no_jugables = [c for c in cookies_originales if c not in cookies_jugables]
        cookies_excluidas.extend(cookies_no_jugables)
        
        # Crear grilla
        grilla = np.zeros((num_filas, num_cols), dtype=int)
        cookies_validas = [c for c in cookies if c.row != -1 and c.col != -1]
        
        # Rellenar grilla (si hay múltiples cookies en la misma celda, usar la más común)
        celda_cookies = {}
        for cookie in cookies_validas:
            key = (cookie.row, cookie.col)
            if key not in celda_cookies:
                celda_cookies[key] = []
            celda_cookies[key].append(cookie)
        
        for (row, col), cookies_en_celda in celda_cookies.items():
            # Si hay múltiples, usar la más frecuente por color
            colores = [c.color for c in cookies_en_celda]
            color_mas_comun = max(set(colores), key=colores.count)
            grilla[row, col] = self.COLOR_MAP[color_mas_comun]
        
        info = {
            "num_filas": num_filas,
            "num_columnas": num_cols,
            "cookies_validas": len(cookies_validas),
            "cookies_excluidas": len(cookies_excluidas),
            "cookies_excluidas_lista": cookies_excluidas,
            "filas_centroids": [c for _, c in filas_ordenadas],
            "columnas_centroids": [c for _, c in columnas_ordenadas],
            "cookies_por_celda": celda_cookies
        }
        
        return grilla, info

    def procesar_imagen(self, imagen_path: str) -> Dict:
        """Función principal de procesamiento."""
        try:
            # Detectar cookies
            cookies = self.detectar_cookies(imagen_path)
            
            if not cookies:
                print("[ERROR] No se detectaron cookies")
                return None
            
            # Construir grilla
            grilla, info = self.construir_grilla_inteligente(cookies)
            
            # Visualizar resultados
            imagen = cv2.imread(imagen_path)
            self._visualizar_resultados(imagen, cookies, grilla, info)
            
            # Imprimir resumen
            self._imprimir_resumen(cookies, grilla, info)
            
            return {
                "cookies": cookies,
                "grilla": grilla,
                "info": info
            }
            
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _visualizar_resultados(self, imagen: np.ndarray, cookies: List[Cookie], 
                               grilla: np.ndarray, info: Dict):
        
        """Visualiza la detección y la grilla."""
        imagen_vis = imagen.copy()

        # Dibujar zona de juego        
        self._dibujar_zona_juego(imagen_vis)
        
        # Dibujar líneas de grilla
        if info.get('filas_centroids') and info.get('columnas_centroids'):
            for y in info['filas_centroids']:
                cv2.line(imagen_vis, 
                    (self.game_area['x_min'], int(y)),
                    (self.game_area['x_max'], int(y)),
                    (255, 255, 0), 1)
                
            for x in info['columnas_centroids']:
                cv2.line(imagen_vis,
                (int(x), self.game_area['y_min']),
                (int(x), self.game_area['y_max']),
                (255, 255, 0), 1)
                    
        # Dibujar cookies válidas
        for cookie in cookies:
            if cookie.row != -1 and cookie.col != -1:
                color = self.colores_dibujo.get(cookie.color, (255, 255, 255))
                cv2.circle(imagen_vis, (cookie.x, cookie.y), 8, color, -1)
                cv2.circle(imagen_vis, (cookie.x, cookie.y), 10, (0, 0, 0), 2)
                
                # Mostrar posición de grilla
                texto = f"{cookie.row},{cookie.col}"
                cv2.putText(imagen_vis, texto, (cookie.x - 15, cookie.y - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Dibujar cookies excluidas
        # check if info has 'cookies_excluidas' key
        if 'cookies_excluidas_lista' in info:
            for cookie in info['cookies_excluidas_lista']:
                cv2.circle(imagen_vis, (cookie.x, cookie.y), 8, (0, 0, 255), 2)
                cv2.putText(imagen_vis, "X", (cookie.x - 5, cookie.y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow("Deteccion con Grilla Inteligente", imagen_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _imprimir_resumen(self, cookies: List[Cookie], grilla: np.ndarray, info: Dict):
        """Imprime resumen de detección."""
        print("\n" + "="*50)
        print("RESUMEN DE DETECCION")
        print("="*50)
        
        # Contar por color
        conteo_colores = {}
        for cookie in cookies:
            if cookie.row != -1:  # Solo contar cookies válidas
                conteo_colores[cookie.color] = conteo_colores.get(cookie.color, 0) + 1
        
        for color, cantidad in conteo_colores.items():
            print(f"{color}: {cantidad} cookies")
        
        print(f"\nCookies validas en grilla: {info['cookies_validas']}")
        print(f"Cookies excluidas: {info['cookies_excluidas']}")
        print(f"Dimension de grilla: {grilla.shape[0]} filas x {grilla.shape[1]} columnas")
        
        if info['cookies_excluidas'] > 0:
            print(f"\nCookies excluidas (outliers):")
            for cookie in info['cookies_excluidas_lista']:
                print(f"   {cookie.color} en ({cookie.x}, {cookie.y})")
        
        print("\n" + "="*50)
        print("GRILLA FINAL")
        print("="*50)
        print("Leyenda: . = Vacio | V = Verde | R = Rojo | A = Amarillo")
        print()
        
        # Crear array con letras
        letra_map = {0: ".", 1: "V", 2: "R", 3: "A"}
        grilla_letras = []
        
        for fila in grilla:
            fila_letras = [letra_map.get(val, "?") for val in fila]
            grilla_letras.append(fila_letras)
        
        # Mostrar grilla con letras
        for i, fila in enumerate(grilla_letras):
            fila_str = " ".join(fila)
            print(f"Fila {i}: {fila_str}")
        
        # Estadísticas de ocupación
        total_celdas = grilla.size
        celdas_ocupadas = np.count_nonzero(grilla)
        ocupacion = (celdas_ocupadas / total_celdas * 100) if total_celdas > 0 else 0
        
        print(f"\nOcupacion: {celdas_ocupadas}/{total_celdas} ({ocupacion:.1f}%)")
        
        # Mostrar grilla numérica original
        print("\nGrilla numerica (para debugging):")
        print(grilla)
        print("="*50 + "\n")

    # Se refactoriza 'imprimir_zona_juego' para recibir la imagen como np.ndarray
    # y solo dibujar el rectángulo, eliminando la carga de imagen y el imshow/waitKey.
    def _dibujar_zona_juego(self, imagen_vis: np.ndarray):
        """
        Dibuja el rectángulo de la zona de juego (self.game_area) en la imagen proporcionada.
        """
        # Obtener las coordenadas del área de juego
        x_min = self.game_area['x_min']
        y_min = self.game_area['y_min']
        x_max = self.game_area['x_max']
        y_max = self.game_area['y_max']

        # Dibujar área de juego
        cv2.rectangle(
            imagen_vis,
            (x_min, y_min),
            (x_max, y_max),
            CONF["game_area_border"]["color"],
            CONF["game_area_border"]["thickness"]
        )

    def return_array_of_images_from_folder(self) -> List[str]:
        """Retorna una lista de rutas de imágenes desde una carpeta dada."""
        imagenes = []
        for archivo in os.listdir(CONF["images_path"]):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                imagenes.append(os.path.join(CONF["images_path"], archivo))
        return imagenes 

def main():
    """Función principal."""
    print("Sistema Mejorado de Deteccion Yoshi's Cookie")
    print("="*50 + "\n")
    
    detector = ImprovedCookieDetector(CONF)

    #resultado = detector.procesar_imagen(CONF["images_path"] + '/001.png')
    #imagenes = os.listdir(CONF["images_path"])
    #imagenes = [img for img in imagenes if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    imagenes = detector.return_array_of_images_from_folder()
    for img_nombre in imagenes:
        print(f"\nProcesando imagen: {img_nombre}")
        resultado = detector.procesar_imagen(img_nombre)
    #if resultado:
    #    print("\n[OK] Procesamiento completado exitosamente")
    #    
    #    # Opcionalmente, analizar movimientos
    #    print("\n¿Deseas analizar movimientos optimos? (requiere movement_analyzer.py)")


if __name__ == '__main__':
    main()

    
