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
        
        # Configuración para filtrar cookies no jugables
        self.CORE_DETECTION_METHOD = "smart_top_exclusion"  # Nuevo método más simple
        self.ALIGNMENT_THRESHOLD = 20  # Tolerancia más permisiva
        self.MIN_ROW_DENSITY = 0.5     # 50% ocupación mínima (más permisivo)
        self.STABLE_ROWS_COUNT = 8     # Tomar más filas desde abajo
        self.MIN_COOKIES_FOR_ALIGNMENT = 3  # Mínimo de cookies para detectar alineación
        self.MIN_FULL_ROWS = 2         # Solo requiere 2 filas completas mínimo
        self.EXCLUDE_TOP_ROWS = 1      # Excluir solo 1 fila superior
        self.MAX_Y_THRESHOLD = 0.25    # Excluir cookies en el 25% superior del área
        
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
        Excluye cookies que están cayendo o no son parte del core jugable.
        """
        # Obtener todas las coordenadas de cookies
        todas_las_coordenadas = []
        for color, data in cookies.items():
            for cx, cy in data["coords"]:
                todas_las_coordenadas.append((cx, cy, color))
        
        # Filtrar cookies no jugables según el método configurado
        cookies_core = self._filtrar_cookies_no_jugables(todas_las_coordenadas)
        
        # Crear grilla vacía
        grilla = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        
        # Calcular dimensiones de cada celda
        area_ancho = self.game_area['x_max'] - self.game_area['x_min']
        area_alto = self.game_area['y_max'] - self.game_area['y_min']
        
        celda_ancho = area_ancho / self.GRID_COLS
        celda_alto = area_alto / self.GRID_ROWS
        
        posiciones_cookies = {}
        cookies_excluidas = []
        
        # Mapear cookies del core a la grilla
        for cx, cy, color in cookies_core:
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
        
        # Identificar cookies excluidas
        for cx, cy, color in todas_las_coordenadas:
            if (cx, cy, color) not in cookies_core:
                cookies_excluidas.append((cx, cy, color))
        
        # Filtrar filas y columnas no jugables
        grilla_filtrada, info_filtrado = self._filtrar_areas_no_jugables(grilla)
        
        return grilla_filtrada, {
            "posiciones_cookies": posiciones_cookies,
            "cookies_excluidas": cookies_excluidas,
            "grilla_original": grilla,
            "info_filtrado": info_filtrado,
            "dimensiones_celda": (celda_ancho, celda_alto),
            "metodo_filtrado": self.CORE_DETECTION_METHOD,
            "cookies_core_count": len(cookies_core),
            "cookies_totales_count": len(todas_las_coordenadas)
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

    def _filtrar_cookies_no_jugables(self, todas_las_coordenadas: List[Tuple]) -> List[Tuple]:
        """
        Filtra cookies que están cayendo o no forman parte del core jugable.
        Implementa diferentes estrategias según CORE_DETECTION_METHOD.
        """
        if self.CORE_DETECTION_METHOD == "alignment":
            return self._filtrar_por_alineacion(todas_las_coordenadas)
        elif self.CORE_DETECTION_METHOD == "density":
            return self._filtrar_por_densidad(todas_las_coordenadas)
        elif self.CORE_DETECTION_METHOD == "stable_rows":
            return self._filtrar_por_filas_estables(todas_las_coordenadas)
        elif self.CORE_DETECTION_METHOD == "hybrid":
            return self._filtrar_hibrido(todas_las_coordenadas)
        elif self.CORE_DETECTION_METHOD == "smart_top_exclusion":
            return self._filtrar_exclusion_inteligente(todas_las_coordenadas)
        else:
            return todas_las_coordenadas  # Sin filtro

    def _filtrar_exclusion_inteligente(self, coordenadas: List[Tuple]) -> List[Tuple]:
        """
        Método más simple y efectivo: excluye solo las cookies del área superior
        basándose en la distribución vertical de cookies.
        """
        if not coordenadas:
            return coordenadas
        
        # Obtener límites del área de juego
        area_alto = self.game_area['y_max'] - self.game_area['y_min']
        umbral_y = self.game_area['y_min'] + (area_alto * self.MAX_Y_THRESHOLD)
        
        # Filtrar cookies que están en la zona superior (probablemente cayendo)
        cookies_core = []
        cookies_excluidas_por_posicion = []
        
        for cx, cy, color in coordenadas:
            if cy > umbral_y:  # Cookie está en la zona jugable (abajo)
                cookies_core.append((cx, cy, color))
            else:  # Cookie está en la zona superior (cayendo)
                cookies_excluidas_por_posicion.append((cx, cy, color))
        
        print(f"🔍 Exclusión inteligente:")
        print(f"   - Umbral Y: {umbral_y}")
        print(f"   - Cookies core: {len(cookies_core)}")
        print(f"   - Cookies excluidas por posición: {len(cookies_excluidas_por_posicion)}")
        
        # Verificación adicional: asegurar que tenemos un core válido
        if len(cookies_core) < 10:  # Si quedan muy pocas cookies, ser más permisivo
            print("⚠️  Muy pocas cookies en core, siendo más permisivo...")
            # Usar un umbral más bajo
            umbral_y_permisivo = self.game_area['y_min'] + (area_alto * 0.15)  # Solo excluir 15% superior
            
            cookies_core = []
            for cx, cy, color in coordenadas:
                if cy > umbral_y_permisivo:
                    cookies_core.append((cx, cy, color))
        
        return cookies_core

    def _filtrar_hibrido(self, coordenadas: List[Tuple]) -> List[Tuple]:
        """
        Método híbrido que combina múltiples estrategias para máxima precisión.
        1. Excluye automáticamente las filas superiores
        2. Filtra por alineación estricta
        3. Verifica densidad por fila
        4. Valida continuidad del core
        """
        if not coordenadas:
            return coordenadas
        
        # Paso 1: Excluir filas superiores automáticamente
        coordenadas_sin_top = self._excluir_filas_superiores(coordenadas)
        
        # Paso 2: Filtrar por alineación
        coordenadas_alineadas = self._filtrar_por_alineacion_estricta(coordenadas_sin_top)
        
        # Paso 3: Verificar densidad y continuidad
        coordenadas_densas = self._filtrar_por_continuidad(coordenadas_alineadas)
        
        return coordenadas_densas

    def _excluir_filas_superiores(self, coordenadas: List[Tuple]) -> List[Tuple]:
        """Excluye automáticamente las N filas superiores."""
        if not coordenadas:
            return coordenadas
        
        # Encontrar los Y únicos y ordenarlos
        ys_unicos = sorted(set(cy for _, cy, _ in coordenadas))
        
        if len(ys_unicos) <= self.EXCLUDE_TOP_ROWS:
            return []  # Si hay muy pocas filas, no excluir nada
        
        # Determinar umbral Y para excluir filas superiores
        y_umbral = ys_unicos[self.EXCLUDE_TOP_ROWS]
        
        # Filtrar coordenadas debajo del umbral
        coordenadas_filtradas = [(cx, cy, color) for cx, cy, color in coordenadas if cy >= y_umbral]
        
        return coordenadas_filtradas

    def _filtrar_por_alineacion_estricta(self, coordenadas: List[Tuple]) -> List[Tuple]:
        """
        Versión más estricta del filtrado por alineación.
        Requiere patrones más regulares y consistentes.
        """
        if len(coordenadas) < self.MIN_COOKIES_FOR_ALIGNMENT:
            return coordenadas
        
        # Agrupar por filas con tolerancia más estricta
        filas_agrupadas = {}
        for cx, cy, color in coordenadas:
            fila_encontrada = None
            for fila_y in filas_agrupadas.keys():
                if abs(cy - fila_y) <= self.ALIGNMENT_THRESHOLD:
                    fila_encontrada = fila_y
                    break
            
            if fila_encontrada is None:
                filas_agrupadas[cy] = [(cx, cy, color)]
            else:
                filas_agrupadas[fila_encontrada].append((cx, cy, color))
        
        # Filtrar filas que no cumplen criterios estrictos
        cookies_validas = []
        filas_completas = 0
        
        for fila_y, cookies_en_fila in filas_agrupadas.items():
            if len(cookies_en_fila) >= self.MIN_COOKIES_FOR_ALIGNMENT:
                # Verificar regularidad del espaciado
                cookies_ordenadas = sorted(cookies_en_fila, key=lambda x: x[0])
                cookies_regulares = self._verificar_regularidad_estricta(cookies_ordenadas)
                
                if len(cookies_regulares) >= self.MIN_COOKIES_FOR_ALIGNMENT:
                    cookies_validas.extend(cookies_regulares)
                    if len(cookies_regulares) >= 5:  # Fila "completa"
                        filas_completas += 1
        
        # Solo devolver cookies si hay suficientes filas completas
        if filas_completas >= self.MIN_FULL_ROWS:
            return cookies_validas
        else:
            return []

    def _verificar_regularidad_estricta(self, cookies_fila: List[Tuple]) -> List[Tuple]:
        """
        Verificación más estricta de regularidad en el espaciado.
        """
        if len(cookies_fila) < 3:
            return cookies_fila
        
        # Calcular distancias entre cookies adyacentes
        distancias = []
        for i in range(1, len(cookies_fila)):
            dist = cookies_fila[i][0] - cookies_fila[i-1][0]
            distancias.append(dist)
        
        if not distancias:
            return cookies_fila
        
        # Verificar consistencia del espaciado
        distancia_promedio = np.mean(distancias)
        desviacion_estandar = np.std(distancias)
        
        # Criterio más estricto: desviación estándar baja
        if desviacion_estandar > distancia_promedio * 0.2:  # 20% de tolerancia
            return []  # Espaciado muy irregular
        
        # Filtrar cookies con espaciado irregular
        tolerancia = distancia_promedio * 0.25
        cookies_regulares = [cookies_fila[0]]
        
        for i in range(1, len(cookies_fila)):
            distancia_actual = cookies_fila[i][0] - cookies_fila[i-1][0]
            if abs(distancia_actual - distancia_promedio) <= tolerancia:
                cookies_regulares.append(cookies_fila[i])
            else:
                break  # Si una cookie está mal alineada, romper secuencia
        
        return cookies_regulares

    def _filtrar_por_continuidad(self, coordenadas: List[Tuple]) -> List[Tuple]:
        """
        Verifica que las cookies formen un bloque continuo desde abajo.
        Elimina cookies sueltas o filas aisladas.
        """
        if not coordenadas:
            return coordenadas
        
        # Agrupar por filas
        filas = {}
        for cx, cy, color in coordenadas:
            if cy not in filas:
                filas[cy] = []
            filas[cy].append((cx, cy, color))
        
        # Ordenar filas de abajo hacia arriba
        filas_ordenadas = sorted(filas.keys(), reverse=True)
        
        # Verificar continuidad desde abajo
        cookies_continuas = []
        fila_anterior = None
        
        for y_actual in filas_ordenadas:
            if fila_anterior is None:
                # Primera fila (más baja), siempre incluir si tiene suficientes cookies
                if len(filas[y_actual]) >= self.MIN_COOKIES_FOR_ALIGNMENT:
                    cookies_continuas.extend(filas[y_actual])
                    fila_anterior = y_actual
            else:
                # Verificar que esta fila sea continua con la anterior
                distancia_filas = fila_anterior - y_actual
                if distancia_filas <= 60:  # Espaciado máximo entre filas (ajustar según tu juego)
                    if len(filas[y_actual]) >= self.MIN_COOKIES_FOR_ALIGNMENT:
                        cookies_continuas.extend(filas[y_actual])
                        fila_anterior = y_actual
                    else:
                        break  # Fila incompleta rompe la continuidad
                else:
                    break  # Gap demasiado grande
        
        return cookies_continuas

    def _filtrar_por_alineacion(self, coordenadas: List[Tuple]) -> List[Tuple]:
        """
        Filtra basándose en alineación horizontal/vertical.
        Las cookies del core deben estar alineadas en filas/columnas regulares.
        """
        if len(coordenadas) < self.MIN_COOKIES_FOR_ALIGNMENT:
            return coordenadas
        
        # Agrupar por filas aproximadas (tolerancia de píxeles)
        filas_agrupadas = {}
        for cx, cy, color in coordenadas:
            # Buscar si ya existe una fila similar
            fila_encontrada = None
            for fila_y in filas_agrupadas.keys():
                if abs(cy - fila_y) <= self.ALIGNMENT_THRESHOLD:
                    fila_encontrada = fila_y
                    break
            
            if fila_encontrada is None:
                filas_agrupadas[cy] = [(cx, cy, color)]
            else:
                filas_agrupadas[fila_encontrada].append((cx, cy, color))
        
        # Filtrar filas con pocas cookies (probablemente cookies cayendo)
        cookies_alineadas = []
        for fila_y, cookies_en_fila in filas_agrupadas.items():
            if len(cookies_en_fila) >= self.MIN_COOKIES_FOR_ALIGNMENT:
                # Verificar alineación horizontal dentro de la fila
                cookies_en_fila.sort(key=lambda x: x[0])  # Ordenar por X
                cookies_bien_alineadas = self._verificar_alineacion_horizontal(cookies_en_fila)
                cookies_alineadas.extend(cookies_bien_alineadas)
        
        return cookies_alineadas

    def _verificar_alineacion_horizontal(self, cookies_fila: List[Tuple]) -> List[Tuple]:
        """
        Verifica que las cookies en una fila estén espaciadas regularmente.
        """
        if len(cookies_fila) < 3:
            return cookies_fila
        
        # Calcular espaciado promedio
        distancias = []
        for i in range(1, len(cookies_fila)):
            dist = cookies_fila[i][0] - cookies_fila[i-1][0]
            distancias.append(dist)
        
        if not distancias:
            return cookies_fila
            
        espaciado_promedio = np.mean(distancias)
        tolerancia_espaciado = espaciado_promedio * 0.3  # 30% de tolerancia
        
        # Filtrar cookies mal alineadas
        cookies_alineadas = [cookies_fila[0]]  # Siempre incluir la primera
        
        for i in range(1, len(cookies_fila)):
            distancia_actual = cookies_fila[i][0] - cookies_fila[i-1][0]
            if abs(distancia_actual - espaciado_promedio) <= tolerancia_espaciado:
                cookies_alineadas.append(cookies_fila[i])
        
        return cookies_alineadas

    def _filtrar_por_densidad(self, coordenadas: List[Tuple]) -> List[Tuple]:
        """
        Filtra basándose en densidad de cookies por área.
        Las áreas con baja densidad probablemente contienen cookies cayendo.
        """
        # Dividir el área de juego en sub-áreas y calcular densidad
        area_ancho = self.game_area['x_max'] - self.game_area['x_min']
        area_alto = self.game_area['y_max'] - self.game_area['y_min']
        
        num_divisiones_x = 4
        num_divisiones_y = 6
        
        sub_area_ancho = area_ancho / num_divisiones_x
        sub_area_alto = area_alto / num_divisiones_y
        
        sub_areas = {}
        
        # Agrupar cookies por sub-área
        for cx, cy, color in coordenadas:
            sub_x = int((cx - self.game_area['x_min']) / sub_area_ancho)
            sub_y = int((cy - self.game_area['y_min']) / sub_area_alto)
            
            sub_x = max(0, min(sub_x, num_divisiones_x - 1))
            sub_y = max(0, min(sub_y, num_divisiones_y - 1))
            
            key = (sub_x, sub_y)
            if key not in sub_areas:
                sub_areas[key] = []
            sub_areas[key].append((cx, cy, color))
        
        # Calcular densidad promedio
        densidades = [len(cookies) for cookies in sub_areas.values()]
        if not densidades:
            return coordenadas
            
        densidad_promedio = np.mean(densidades)
        umbral_densidad = densidad_promedio * self.MIN_ROW_DENSITY
        
        # Filtrar cookies en áreas de baja densidad
        cookies_densas = []
        for key, cookies in sub_areas.items():
            if len(cookies) >= umbral_densidad:
                cookies_densas.extend(cookies)
        
        return cookies_densas

    def _filtrar_por_filas_estables(self, coordenadas: List[Tuple]) -> List[Tuple]:
        """
        Toma solo las cookies de las filas más bajas (estables).
        Asume que las cookies cayendo están en la parte superior.
        """
        if not coordenadas:
            return coordenadas
        
        # Ordenar por coordenada Y (de arriba hacia abajo)
        coordenadas_ordenadas = sorted(coordenadas, key=lambda x: x[1], reverse=True)
        
        # Agrupar por filas aproximadas
        filas = {}
        for cx, cy, color in coordenadas_ordenadas:
            # Buscar fila existente dentro del umbral
            fila_key = None
            for y_existente in filas.keys():
                if abs(cy - y_existente) <= self.ALIGNMENT_THRESHOLD:
                    fila_key = y_existente
                    break
            
            if fila_key is None:
                fila_key = cy
                filas[fila_key] = []
            
            filas[fila_key].append((cx, cy, color))
        
        # Ordenar filas por Y (más abajo primero)
        filas_ordenadas = sorted(filas.items(), key=lambda x: x[0], reverse=True)
        
        # Tomar solo las filas más estables (las de abajo)
        cookies_estables = []
        filas_tomadas = 0
        
        for y_fila, cookies_en_fila in filas_ordenadas:
            if len(cookies_en_fila) >= self.MIN_COOKIES_FOR_ALIGNMENT:
                cookies_estables.extend(cookies_en_fila)
                filas_tomadas += 1
                
                if filas_tomadas >= self.STABLE_ROWS_COUNT:
                    break
        
        return cookies_estables

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
        print(f"🎯 Cookies en el core jugable: {info_grilla['cookies_core_count']}")
        print(f"🚫 Cookies excluidas (cayendo): {info_grilla['cookies_totales_count'] - info_grilla['cookies_core_count']}")
        print(f"🔧 Método de filtrado: {info_grilla['metodo_filtrado']}")
        print(f"📐 Dimensión grilla original: {info_grilla['info_filtrado']['dimension_original']}")
        print(f"📐 Dimensión grilla filtrada: {info_grilla['info_filtrado']['dimension_filtrada']}")
        print(f"🗑️ Filas eliminadas: {info_grilla['info_filtrado']['filas_eliminadas']}")
        print(f"🗑️ Columnas eliminadas: {info_grilla['info_filtrado']['columnas_eliminadas']}")
        
        if info_grilla['cookies_excluidas']:
            print(f"\n--- Cookies Excluidas (Coordenadas) ---")
            for cx, cy, color in info_grilla['cookies_excluidas']:
                print(f"❌ {color}: ({cx}, {cy})")
        
        print("\n--- Grilla 2D Final ---")
        print("(1=Verde, 2=Rojo, 3=Amarillo, 0=Vacío)")
        print(grilla_2d)


def main():
    # Método simplificado pero efectivo
    detector = CookieDetector(CONF)
    
    # Configuración más simple y efectiva
    detector.CORE_DETECTION_METHOD = "smart_top_exclusion"  # Método más simple
    detector.MAX_Y_THRESHOLD = 0.25  # Excluir cookies en el 25% superior del área
    
    # Si quieres probar otros métodos:
    # detector.CORE_DETECTION_METHOD = "stable_rows"  # Tomar solo filas inferiores
    # detector.STABLE_ROWS_COUNT = 7                  # Número de filas desde abajo
    
    resultado = detector.procesar_imagen('imgs/static_image.jpg')
    
    if resultado is not None:
        imagen_con_detecciones = resultado["imagen_con_detecciones"].copy()
        
        # Dibujar línea de separación donde se hace el corte
        area_alto = detector.game_area['y_max'] - detector.game_area['y_min']
        umbral_y = int(detector.game_area['y_min'] + (area_alto * detector.MAX_Y_THRESHOLD))
        
        cv2.line(imagen_con_detecciones, 
                (detector.game_area['x_min'], umbral_y), 
                (detector.game_area['x_max'], umbral_y), 
                (255, 0, 255), 3)  # Línea magenta
        
        cv2.putText(imagen_con_detecciones, "CORTE AQUI", 
                   (detector.game_area['x_min'] + 10, umbral_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Marcar cookies excluidas
        for cx, cy, color in resultado["info_grilla"]["cookies_excluidas"]:
            cv2.circle(imagen_con_detecciones, (cx, cy), 8, (0, 0, 255), 2)
            cv2.putText(imagen_con_detecciones, "X", (cx-5, cy+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow("Filtrado Inteligente (Magenta=Corte, Rojo=Excluidas)", imagen_con_detecciones)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        grilla_2d = resultado["grilla_2d"]
        print(f"\n=== RESULTADO FINAL ===")
        print(f"Grilla 2D shape: {grilla_2d.shape}")
        print(f"Core jugable contiene {np.count_nonzero(grilla_2d)} cookies")
        
        # Mostrar la grilla de forma más visual
        if grilla_2d.size > 0:
            print("\n=== GRILLA VISUAL ===")
            color_symbols = {0: "⚫", 1: "🟢", 2: "🔴", 3: "🟡"}
            for i, fila in enumerate(grilla_2d):
                fila_visual = " ".join([color_symbols.get(val, "❓") for val in fila])
                print(f"Fila {i}: {fila_visual}")
        else:
            print("❌ Grilla vacía - ajustar parámetros")
            
        # Sugerencias de ajuste
        if resultado["info_grilla"]["cookies_core_count"] == 0:
            print("\n💡 SUGERENCIAS:")
            print("1. Aumentar MAX_Y_THRESHOLD (ej: 0.35)")
            print("2. Probar método 'stable_rows'")
            print("3. Verificar que el área de juego esté bien definida")

if __name__ == '__main__':
    main()