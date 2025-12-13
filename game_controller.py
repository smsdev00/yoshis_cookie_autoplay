"""
Controlador principal del juego Yoshi's Cookie.
Captura pantalla, detecta cookies, calcula mejor movimiento y ejecuta acciones.
"""

import pyautogui
import numpy as np
import cv2
import time
from typing import Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime


class GameController:
    """Controlador principal del juego."""
    
    def __init__(self, game_window_region: Tuple[int, int, int, int] = None):
        """
        Args:
            game_window_region: (x, y, width, height) de la ventana del juego.
                               Si es None, se debe configurar manualmente.
        """
        self.game_window_region = game_window_region
        self.screenshots_dir = Path("./screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Configuración de pyautogui
        pyautogui.PAUSE = 0.1  # Pausa entre comandos
        pyautogui.FAILSAFE = True  # Mover mouse a esquina superior izquierda para detener
        
        # Estado del juego
        self.current_screenshot = None
        self.move_history = []
        
        print("[GameController] Inicializado")
        print(f"[INFO] Failsafe activado: mueve el mouse a la esquina superior izquierda para detener")

    def set_game_window(self, x: int, y: int, width: int, height: int):
        """Configura la región de la ventana del juego."""
        self.game_window_region = (x, y, width, height)
        print(f"[GameController] Ventana configurada: x={x}, y={y}, w={width}, h={height}")

    def capture_screenshot(self, save: bool = True) -> np.ndarray:
        """
        Captura screenshot del área del juego.
        
        Args:
            save: Si True, guarda la imagen en disco
            
        Returns:
            Imagen en formato numpy array (BGR)
        """
        if self.game_window_region is None:
            raise ValueError("Debes configurar game_window_region primero con set_game_window()")
        
        x, y, width, height = self.game_window_region
        
        # Capturar con pyautogui (devuelve PIL Image)
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        
        # Convertir PIL a numpy array (RGB)
        img_rgb = np.array(screenshot)
        
        # Convertir RGB a BGR (formato OpenCV)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        self.current_screenshot = img_bgr
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = self.screenshots_dir / f"game_{timestamp}.jpg"
            cv2.imwrite(str(filename), img_bgr)
            print(f"[GameController] Screenshot guardado: {filename}")
        
        return img_bgr

    def execute_move(self, move: Dict, game_area_offset: Tuple[int, int] = None):
        """
        Ejecuta un movimiento en el juego.
        
        Args:
            move: Diccionario con información del movimiento
            game_area_offset: (offset_x, offset_y) del área de juego respecto a la ventana capturada
        """
        if move is None:
            print("[GameController] No hay movimiento para ejecutar")
            return
        
        # Por ahora, solo logueamos el movimiento
        # En la siguiente iteración implementaremos la ejecución real
        print(f"[GameController] Movimiento a ejecutar: {move}")
        self.move_history.append({
            'timestamp': datetime.now(),
            'move': move
        })

    def wait_for_animation(self, duration: float = 1.0):
        """Espera a que termine la animación del juego."""
        print(f"[GameController] Esperando {duration}s por animación...")
        time.sleep(duration)

    def is_game_over(self) -> bool:
        """
        Detecta si el juego terminó.
        Implementación básica - puede mejorarse con OCR o detección de imágenes.
        """
        # TODO: Implementar detección real de game over
        return False

    def get_move_history(self) -> list:
        """Retorna el historial de movimientos."""
        return self.move_history

    def clear_history(self):
        """Limpia el historial de movimientos."""
        self.move_history = []
        print("[GameController] Historial limpiado")


class GameWindowFinder:
    """Herramienta para encontrar y configurar la ventana del juego."""
    
    @staticmethod
    def find_window_interactive():
        """
        Modo interactivo para encontrar la ventana del juego.
        El usuario debe hacer clic en dos esquinas.
        """
        print("\n" + "="*50)
        print("CONFIGURACION DE VENTANA DEL JUEGO")
        print("="*50)
        print("\nInstrucciones:")
        print("1. Asegurate de que la ventana del juego este visible")
        print("2. Presiona ENTER para empezar")
        print("3. Haz clic en la esquina SUPERIOR IZQUIERDA del area de juego")
        print("4. Haz clic en la esquina INFERIOR DERECHA del area de juego")
        print("\nPresiona ENTER para continuar...")
        input()
        
        print("\n[INFO] Haz clic en la esquina SUPERIOR IZQUIERDA...")
        time.sleep(1)
        
        # Esperar primer clic
        print("[INFO] Esperando clic...")
        x1, y1 = pyautogui.position()
        print(f"[OK] Esquina superior izquierda: ({x1}, {y1})")
        
        print("\n[INFO] Ahora haz clic en la esquina INFERIOR DERECHA...")
        time.sleep(2)
        
        # Esperar segundo clic
        x2, y2 = pyautogui.position()
        print(f"[OK] Esquina inferior derecha: ({x2}, {y2})")
        
        width = x2 - x1
        height = y2 - y1
        
        print(f"\n[OK] Ventana configurada:")
        print(f"    X: {x1}")
        print(f"    Y: {y1}")
        print(f"    Ancho: {width}")
        print(f"    Alto: {height}")
        
        return (x1, y1, width, height)
    
    @staticmethod
    def get_current_mouse_position():
        """Obtiene la posición actual del mouse (útil para debugging)."""
        x, y = pyautogui.position()
        print(f"[INFO] Posicion del mouse: ({x}, {y})")
        return (x, y)
    
    @staticmethod
    def show_screen_info():
        """Muestra información de la pantalla."""
        screen_width, screen_height = pyautogui.size()
        print(f"[INFO] Resolucion de pantalla: {screen_width}x{screen_height}")
        return (screen_width, screen_height)


def test_game_controller():
    """Función de prueba para el controlador."""
    print("\n" + "="*50)
    print("TEST: Game Controller")
    print("="*50)
    
    # Mostrar info de pantalla
    GameWindowFinder.show_screen_info()
    
    # Crear controlador
    controller = GameController()
    
    # Opción 1: Configurar manualmente
    # controller.set_game_window(x=100, y=100, width=800, height=600)
    
    # Opción 2: Configurar interactivamente
    print("\n¿Quieres configurar la ventana interactivamente? (s/n): ", end="")
    respuesta = input().strip().lower()
    
    if respuesta == 's':
        region = GameWindowFinder.find_window_interactive()
        controller.set_game_window(*region)
        
        # Capturar screenshot de prueba
        print("\n[INFO] Capturando screenshot de prueba...")
        img = controller.capture_screenshot(save=True)
        print(f"[OK] Screenshot capturado: {img.shape}")
        
        # Mostrar la imagen
        cv2.imshow("Screenshot de prueba", img)
        print("[INFO] Presiona cualquier tecla en la ventana de imagen para continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("[INFO] Configuracion manual requerida")
    
    print("\n[OK] Test completado")


if __name__ == "__main__":
    test_game_controller()
