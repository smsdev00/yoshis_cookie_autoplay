import cv2
import numpy as np

def detect_cookies_by_color_improved():
    # Cargar la imagen
    image = cv2.imread('imgs/static_image.jpg')
    
    if image is None:
        print("❌ Error: No se pudo cargar la imagen. Verifica la ruta.")
        return []

    # Convertir a HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 🔧 RANGOS MEJORADOS PARA VERDE (más amplios)
    # Rango 1: Verde principal
    lower_green1 = np.array([40, 50, 50])    # HSV más permisivo
    upper_green1 = np.array([80, 255, 255])
    
    # Rango 2: Verde más claro/amarillento
    lower_green2 = np.array([35, 40, 80])
    upper_green2 = np.array([75, 180, 255])
    
    # Crear máscaras para ambos rangos
    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    
    # Combinar máscaras
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 🔧 MORFOLOGÍA MÁS AGRESIVA
    kernel = np.ones((5,5), np.uint8)  # Kernel más grande
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Filtro Gaussiano para suavizar
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = []
    valid_contours = []
    
    # 🎯 DEFINIR ÁREA DE JUEGO (ajustado a tu imagen)
    game_area = {
        'x_min': 345, 'x_max': 525,  # Área horizontal de la cuadrícula ajustada
        'y_min': 145, 'y_max': 295   # Área vertical de la cuadrícula ajustada
    }
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 🔧 FILTROS MÁS PERMISIVOS (el problema principal)
        # Área mucho más permisiva
        if area < 30 or area > 1000:  # Muy amplio para no perder detecciones
            continue
            
        # Calcular centro primero para filtrar por ubicación
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # 🚫 FILTRAR POR ÁREA DE JUEGO
        if not (game_area['x_min'] <= cx <= game_area['x_max'] and 
                game_area['y_min'] <= cy <= game_area['y_max']):
            continue
        
        # 🔍 FILTRO DE PROXIMIDAD (evitar duplicados)
        too_close = False
        min_distance = 20  # Distancia reducida
        for existing_center in centers:
            distance = np.sqrt((cx - existing_center[0])**2 + (cy - existing_center[1])**2)
            if distance < min_distance:
                too_close = True
                break
        
        if too_close:
            continue
            
        # ⚠️ COMENTAMOS FILTROS ESTRICTOS TEMPORALMENTE
        # Aproximar contorno para reducir ruido
        # epsilon = 0.02 * cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # # Filtro por circularidad mejorado
        # perimeter = cv2.arcLength(cnt, True)
        # if perimeter > 0:
        #     circularity = 4 * np.pi * area / (perimeter * perimeter)
        #     if circularity < 0.3:  # Muy permisivo
        #         continue
        
        # # Filtro por aspect ratio (relación ancho/alto)
        # x, y, w, h = cv2.boundingRect(cnt)
        # aspect_ratio = float(w) / h
        # if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # Muy permisivo
        #     continue
            
        centers.append((cx, cy))
        valid_contours.append(cnt)
    
    # 📊 MOSTRAR INFORMACIÓN DE DEBUG DETALLADA
    print(f"✅ Total de contornos encontrados: {len(contours)}")
    print(f"✅ Contornos válidos después de filtros: {len(valid_contours)}")
    print(f"✅ Se detectaron {len(centers)} cookies verdes.")
    if centers:
        print("📍 Centros:", centers)
    else:
        print("❌ No se detectaron cookies. Analizando contornos...")
        
        # Debug: analizar primeros 10 contornos
        for i, cnt in enumerate(contours[:10]):
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                in_game_area = (game_area['x_min'] <= cx <= game_area['x_max'] and 
                              game_area['y_min'] <= cy <= game_area['y_max'])
                print(f"  Contorno {i+1}: área={area:.0f}, centro=({cx},{cy}), en_área_juego={in_game_area}")
        
        print(f"\n🎯 Área de juego definida: x={game_area['x_min']}-{game_area['x_max']}, y={game_area['y_min']}-{game_area['y_max']}")

    # 🎨 VISUALIZACIÓN MEJORADA
    image_copy = image.copy()
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Dibujar área de juego
    cv2.rectangle(image_copy, 
                 (game_area['x_min'], game_area['y_min']), 
                 (game_area['x_max'], game_area['y_max']), 
                 (255, 255, 0), 2)  # Rectángulo azul para área de juego
    
    # Dibujar contornos válidos
    cv2.drawContours(image_copy, valid_contours, -1, (255, 0, 255), 2)  # Magenta
    
    # Dibujar centros con mejor visualización
    for i, (x, y) in enumerate(centers):
        cv2.circle(image_copy, (x, y), 10, (0, 255, 0), -1)  # Verde relleno más grande
        cv2.circle(image_copy, (x, y), 15, (0, 0, 255), 2)   # Borde rojo más grande
        # Numerar las detecciones con fondo
        cv2.putText(image_copy, str(i+1), (x-8, y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image_copy, str(i+1), (x-8, y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Mostrar imágenes lado a lado
    combined = np.hstack([image_copy, mask_colored])
    
    # Redimensionar si es muy grande
    height, width = combined.shape[:2]
    if width > 1200:
        scale = 1200 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        combined = cv2.resize(combined, (new_width, new_height))
    
    cv2.imshow("Detección de cookies verdes + Máscara", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return centers

def test_different_ranges():
    """Función para probar diferentes rangos de color"""
    image = cv2.imread('imgs/static_image.jpg')
    if image is None:
        return
        
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Diferentes rangos para probar
    ranges = [
        ("Verde estándar", [40, 50, 50], [80, 255, 255]),
        ("Verde amplio", [35, 30, 30], [85, 255, 255]),
        ("Verde-amarillo", [30, 40, 60], [90, 200, 255]),
    ]
    
    for name, lower, upper in ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"{name}: {len(contours)} contornos encontrados")
        
        cv2.imshow(f"Máscara - {name}", mask)
        cv2.waitKey(1000)  # Mostrar por 1 segundo
    
    cv2.destroyAllWindows()

# 🚀 EJECUTAR
if __name__ == "__main__":
    print("🔍 Probando diferentes rangos...")
    test_different_ranges()
    
    print("\n🎯 Ejecutando detección mejorada...")
    centers = detect_cookies_by_color_improved()
    
    if len(centers) == 0:
        print("\n💡 SUGERENCIAS:")
        print("1. Verifica que la imagen esté en 'imgs/static_image.jpg'")
        print("2. Las cookies verdes pueden tener mucho amarillo mezclado")
        print("3. Prueba ajustar los rangos HSV manualmente")
        print("4. Considera usar detección por template matching")