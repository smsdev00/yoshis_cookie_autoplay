import cv2
import numpy as np
from config import CONF
from sklearn.cluster import DBSCAN


class ImprovedCookieDetector:
    def __init__(self, config):
        self.config = config
        self.cookie_colors_hsv = {
            name: {"min": np.array(values["min"]),
                   "max": np.array(values["max"])}
            for name, values in config["cookies_colors"].items()
        }
        self.color_map = {
            "Rojo": 1,
            "Verde": 2,
            "Amarillo": 3,
            "Azul": 4,
            "Marrón": 5,
            # Un sexto color si es necesario
        }

    def detectar_cookies(self, image_path, roi=None):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo cargar la imagen en {image_path}")
            return []

        if roi:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        all_cookies = []

        for color_name, hsv_range in self.cookie_colors_hsv.items():
            mask = cv2.inRange(hsv, hsv_range["min"], hsv_range["max"])
            
            # Limpieza de la máscara
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:  # Umbral de área para no detectar ruido
                    x, y, w, h = cv2.boundingRect(cnt)
                    all_cookies.append({
                        "center": (x + w // 2, y + h // 2),
                        "color_name": color_name,
                        "color_id": self.color_map.get(color_name, 0),
                        "area": area
                    })
        
        return all_cookies

    def construir_grilla_inteligente(self, cookies, grid_shape=(6, 5), tolerance=35):
        if not cookies:
            return np.zeros(grid_shape, dtype=int), {}

        positions = np.array([c["center"] for c in cookies])
        
        # Clustering para identificar filas y columnas
        db_y = DBSCAN(eps=tolerance, min_samples=1).fit(positions[:, 1].reshape(-1, 1))
        row_labels = db_y.labels_
        
        db_x = DBSCAN(eps=tolerance, min_samples=1).fit(positions[:, 0].reshape(-1, 1))
        col_labels = db_x.labels_

        unique_rows = np.unique(row_labels)
        unique_cols = np.unique(col_labels)

        # Mapeo de clusters a índices de grilla
        row_map = {label: i for i, label in enumerate(np.sort(unique_rows))}
        col_map = {label: i for i, label in enumerate(np.sort(unique_cols))}
        
        num_rows = len(unique_rows)
        num_cols = len(unique_cols)

        # Si el clustering no da el tamaño esperado, ajustamos
        if num_rows != grid_shape[0] or num_cols != grid_shape[1]:
            print(f"Warning: El clustering detectó {num_rows}x{num_cols} en lugar de {grid_shape[0]}x{grid_shape[1]}. Se intentará ajustar.")
            # Un enfoque más robusto podría ser necesario aquí, como k-means con n_clusters conocido
            
        grid = np.zeros(grid_shape, dtype=int)
        
        for i, cookie in enumerate(cookies):
            row_idx_cluster = row_labels[i]
            col_idx_cluster = col_labels[i]
            
            row = row_map.get(row_idx_cluster)
            col = col_map.get(col_idx_cluster)

            if row is not None and col is not None and row < grid_shape[0] and col < grid_shape[1]:
                grid[row, col] = cookie["color_id"]

        info = {
            "num_cookies": len(cookies),
            "detected_shape": (num_rows, num_cols),
            "row_map": row_map,
            "col_map": col_map
        }
        
        return grid, info


if __name__ == '__main__':
    # Prueba de detección en una imagen estática
    detector = ImprovedCookieDetector(CONF)
    
    # La imagen debe existir en la ruta especificada
    image_to_test = CONF["images_path"] + "static_image.jpg"
    
    print(f"Probando detección en: {image_to_test}")
    
    # ROI (Region of Interest) opcional, si se conoce
    # game_area = CONF.get("game_area")
    # roi = (game_area['x_min'], game_area['y_min'], game_area['x_max'] - game_area['x_min'], game_area['y_max'] - game_area['y_min'])
    
    detected_cookies = detector.detectar_cookies(image_to_test)
    
    if detected_cookies:
        print(f"Se detectaron {len(detected_cookies)} cookies.")
        
        # Tamaño de la grilla para Yoshi's Cookie suele ser 6 filas x 5 columnas
        grilla, info = detector.construir_grilla_inteligente(detected_cookies, grid_shape=(6, 5), tolerance=40)
        
        print("\nGrilla construida:")
        print(grilla)
        print("\nInformación de la grilla:")
        print(info)
        
        # Visualización (opcional)
        image = cv2.imread(image_to_test)
        for cookie in detected_cookies:
            center = cookie['center']
            cv2.circle(image, center, 15, (0, 255, 0), 2)
            cv2.putText(image, str(cookie['color_id']), (center[0] - 5, center[1] + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        # cv2.imshow("Detección", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print("\nVisualización desactivada. Descomenta el código para ver la imagen.")

    else:
        print("No se detectaron cookies. Revisa la configuración de colores y la imagen.")