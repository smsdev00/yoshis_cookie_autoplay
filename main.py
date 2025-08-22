from config import CONF
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

class CookieDetector:
    def __init__(self, config: dict):
        self.config = config
        self.cookies_colors = config["cookies_colors"]
        self.game_area = config["game_area"]
        self.images_path = config["images_path"]
        
        # Configuración de la grilla (ajustar según tu juego)
        self.GRID_ROWS = 10  # Número de filas del tablero
        self.GRID_COLS = 8   # Número de columnas del tablero
        self.MIN_COOKIES_PER_ROW = 3  # Mínimo de cookies para considerar fila válida
        self.MIN_COOKIES_PER_COL = 3  # Mínimo de cookies para considerar columna válida
        
        # Mapeo de colores a valores numéricos
        self.COLOR_MAP = {
            "Verde": 1,
            "Rojo": 2, 
            "Amarillo": 3,
            "": 0  # Celda vacía
        }
        
        self.colores_dibujo = {
            'Verde': (235, 20, 160),
            'Rojo': (255, 255, 255),
            'Amarillo': (0, 255, 255),
        }

    def detectar_cookies(self, imagen_path: str) -> Dict:
        """Detecta cookies y devuelve sus coordenadas organizadas por color."""
        cookies = {color: {"cant": 0, "coords": []} for color in self.cookies_colors.keys()}
        
        imagen = cv2.imread(imagen_path)
        if imagen is None:
            raise ValueError(f"Error: No se pudo cargar la imagen desde {imagen_path}")

        imagen_con_detecciones = imagen.copy()
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        # Dibujar área de juego
        cv2.rectangle(
            imagen_con_detecciones,
            (self.game_area['x_min'], self.game_area['y_min']),
            (self.game_area['x_max'], self.game_area['y_max']),
            (255, 0, 0), 2
        )

        for nombre_color, rangos in self.cookies_colors.items():
            contornos_validos = self._filtrar_contornos(hsv, rangos, nombre_color)
            
            for contorno in contornos_validos:
                cx, cy = self._obtener_centro(contorno)
                
                if self._esta_en_area_juego(cx, cy):
                    cookies[nombre_color]["cant"] += 1
                    cookies[nombre_color]["coords"].append((cx, cy))
                    
                    # Visualizar detección
                    color_punto = self.colores_dibujo.get(nombre_color, (255, 255, 255))
                    cv2.circle(imagen_con_detecciones, (cx, cy), 5, color_punto, -1)
                    print(f"✅ {nombre_color} cookie detectada en: ({cx}, {cy})")

        return {
            "cookies": cookies,
            "imagen_con_detecciones": imagen_con_detecciones
        }

    def _filtrar_contornos(self, hsv: np.ndarray, rangos: dict, nombre_color: str) -> List:
        """Filtra contornos por color y forma según el tipo de cookie."""
        mascara = cv2.inRange(hsv, np.array(rangos['min']), np.array(rangos['max']))
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contornos_validos = []
        
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area <= 600:  # Filtro por área mínima
                continue
                
            # Filtros específicos por color
            if nombre_color == 'Verde':
                if not self._es_forma_circular(contorno):
                    continue
                    
            if not self._tiene_aspect_ratio_valido(contorno):
                continue
                
            contornos_validos.append(contorno)
            
        return contornos_validos

    def _es_forma_circular(self, contorno) -> bool:
        """Verifica si un contorno tiene forma aproximadamente circular."""
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        if perimetro == 0:
            return False
        circularidad = 4 * np.pi * area / (perimetro * perimetro)
        return circularidad >= 0.6

    def _tiene_aspect_ratio_valido(self, contorno) -> bool:
        """Verifica si el aspect ratio del contorno es válido."""
        x, y, w, h = cv2.boundingRect(contorno)
        aspect_ratio = float(w) / h
        return 0.5 <= aspect_ratio <= 2.0

    def _obtener_centro(self, contorno) -> Tuple[int, int]:
        """Calcula el centro de masa de un contorno."""
        momentos = cv2.moments(contorno)
        if momentos["m00"] == 0:
            return (0, 0)
        cx = int(momentos["m10"] / momentos["m00"])
        cy = int(momentos["m01"] / momentos["m00"])
        return cx, cy

    def _esta_en_area_juego(self, x: int, y: int) -> bool:
        """Verifica si las coordenadas están dentro del área de juego."""
        return (self.game_area['x_min'] <= x <= self.game_area['x_max'] and
                self.game_area['y_min'] <= y <= self.game_area['y_max'])

    def coordenadas_a_grilla(self, cookies: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Convierte las coordenadas de cookies a una grilla 2D.
        Descarta filas/columnas con pocas cookies (no jugables).
        """
        # Crear grilla vacía
        grilla = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        
        # Calcular dimensiones de cada celda
        area_ancho = self.game_area['x_max'] - self.game_area['x_min']
        area_alto = self.game_area['y_max'] - self.game_area['y_min']
        
        celda_ancho = area_ancho / self.GRID_COLS
        celda_alto = area_alto / self.GRID_ROWS
        
        posiciones_cookies = {}
        
        # Mapear cada cookie a su posición en la grilla
        for color, data in cookies.items():
            for cx, cy in data["coords"]:
                # Convertir coordenadas absolutas a posición de grilla
                col = int((cx - self.game_area['x_min']) / celda_ancho)
                fila = int((cy - self.game_area['y_min']) / celda_alto)
                
                # Asegurar que estén dentro de los límites
                col = max(0, min(col, self.GRID_COLS - 1))
                fila = max(0, min(fila, self.GRID_ROWS - 1))
                
                # Asignar valor del color a la grilla
                grilla[fila, col] = self.COLOR_MAP[color]
                posiciones_cookies[(fila, col)] = {
                    "color": color,
                    "coordenadas": (cx, cy)
                }
        
        # Filtrar filas y columnas no jugables
        grilla_filtrada, info_filtrado = self._filtrar_areas_no_jugables(grilla)
        
        return grilla_filtrada, {
            "posiciones_cookies": posiciones_cookies,
            "grilla_original": grilla,
            "info_filtrado": info_filtrado,
            "dimensiones_celda": (celda_ancho, celda_alto)
        }

    def _filtrar_areas_no_jugables(self, grilla: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Elimina filas y columnas que no tienen suficientes cookies (áreas no jugables).
        """
        filas_validas = []
        columnas_validas = []
        
        # Identificar filas válidas
        for i, fila in enumerate(grilla):
            cookies_en_fila = np.count_nonzero(fila)
            if cookies_en_fila >= self.MIN_COOKIES_PER_ROW:
                filas_validas.append(i)
        
        # Identificar columnas válidas
        for j in range(grilla.shape[1]):
            cookies_en_columna = np.count_nonzero(grilla[:, j])
            if cookies_en_columna >= self.MIN_COOKIES_PER_COL:
                columnas_validas.append(j)
        
        # Crear grilla filtrada
        if filas_validas and columnas_validas:
            grilla_filtrada = grilla[np.ix_(filas_validas, columnas_validas)]
        else:
            grilla_filtrada = np.array([[]])
        
        info_filtrado = {
            "filas_eliminadas": [i for i in range(grilla.shape[0]) if i not in filas_validas],
            "columnas_eliminadas": [j for j in range(grilla.shape[1]) if j not in columnas_validas],
            "filas_validas": filas_validas,
            "columnas_validas": columnas_validas,
            "dimension_original": grilla.shape,
            "dimension_filtrada": grilla_filtrada.shape
        }
        
        return grilla_filtrada, info_filtrado

    def procesar_imagen(self, imagen_path: str) -> Dict:
        """Función principal que procesa una imagen y devuelve la grilla 2D."""
        try:
            # Detectar cookies
            resultado_deteccion = self.detectar_cookies(imagen_path)
            cookies = resultado_deteccion["cookies"]
            imagen_con_detecciones = resultado_deteccion["imagen_con_detecciones"]
            
            # Convertir a grilla 2D
            grilla_2d, info_grilla = self.coordenadas_a_grilla(cookies)
            
            # Imprimir resumen
            self._imprimir_resumen(cookies, grilla_2d, info_grilla)
            
            return {
                "cookies": cookies,
                "grilla_2d": grilla_2d,
                "info_grilla": info_grilla,
                "imagen_con_detecciones": imagen_con_detecciones
            }
            
        except Exception as e:
            print(f"❌ Error procesando imagen: {e}")
            return None

    def _imprimir_resumen(self, cookies: Dict, grilla_2d: np.ndarray, info_grilla: Dict):
        """Imprime un resumen de la detección y el procesamiento."""
        print("\n--- Resumen de Detecciones ---")
        total_cookies = 0
        for nombre, datos in cookies.items():
            print(f"🍪 {nombre}: {datos['cant']} cookies")
            total_cookies += datos['cant']
        
        print(f"\n✅ Total de cookies detectadas: {total_cookies}")
        print(f"📐 Dimensión grilla original: {info_grilla['info_filtrado']['dimension_original']}")
        print(f"📐 Dimensión grilla filtrada: {info_grilla['info_filtrado']['dimension_filtrada']}")
        print(f"🗑️ Filas eliminadas: {info_grilla['info_filtrado']['filas_eliminadas']}")
        print(f"🗑️ Columnas eliminadas: {info_grilla['info_filtrado']['columnas_eliminadas']}")
        
        print("\n--- Grilla 2D Final ---")
        print("(1=Verde, 2=Rojo, 3=Amarillo, 0=Vacío)")
        print(grilla_2d)


def main():
    detector = CookieDetector(CONF)
    resultado = detector.procesar_imagen('imgs/static_image.jpg')
    
    if resultado is not None:
        # Mostrar imagen con detecciones
        cv2.imshow("Detección de Cookies", resultado["imagen_con_detecciones"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # La grilla 2D está disponible en resultado["grilla_2d"]
        grilla_2d = resultado["grilla_2d"]
        print(f"\nGrilla 2D shape: {grilla_2d.shape}")

if __name__ == '__main__':
    main()