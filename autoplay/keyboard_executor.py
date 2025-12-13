"""
Módulo para ejecutar movimientos físicos en el juego usando teclado/mouse.
Traduce las posiciones de la grilla a comandos del teclado.
"""

import pyautogui
import time
from typing import Tuple
from enum import Enum


class InputMethod(Enum):
    """Métodos de input disponibles."""
    KEYBOARD_ARROWS = "keyboard_arrows"  # Flechas + Enter/Espacio
    MOUSE_CLICK = "mouse_click"  # Click directo en las cookies
    KEYBOARD_WASD = "keyboard_wasd"  # WASD + Espacio


class KeyboardExecutor:
    """Ejecuta movimientos usando el teclado."""
    
    def __init__(self, input_method: InputMethod = InputMethod.KEYBOARD_ARROWS):
        """
        Args:
            input_method: Método de input a usar
        """
        self.input_method = input_method
        self.key_delay = 0.05  # Delay entre teclas (segundos)
        self.action_delay = 0.1  # Delay después de acciones importantes
        
        # Posición actual del cursor en la grilla
        self.cursor_position = (0, 0)  # (fila, columna)
        
        print(f"[KeyboardExecutor] Inicializado con metodo: {input_method.value}")

    def reset_cursor_position(self):
        """Resetea la posición del cursor a (0, 0)."""
        self.cursor_position = (0, 0)
        print("[KeyboardExecutor] Cursor reseteado a (0, 0)")

    def move_cursor_to(self, target_row: int, target_col: int):
        """
        Mueve el cursor a la posición objetivo usando flechas.
        
        Args:
            target_row: Fila objetivo (0-indexed)
            target_col: Columna objetivo (0-indexed)
        """
        current_row, current_col = self.cursor_position
        
        # Calcular movimientos necesarios
        row_diff = target_row - current_row
        col_diff = target_col - current_col
        
        print(f"[KeyboardExecutor] Moviendo cursor de ({current_row},{current_col}) a ({target_row},{target_col})")
        
        # Movimiento vertical
        if row_diff > 0:
            # Mover hacia abajo
            for _ in range(abs(row_diff)):
                pyautogui.press('down')
                time.sleep(self.key_delay)
        elif row_diff < 0:
            # Mover hacia arriba
            for _ in range(abs(row_diff)):
                pyautogui.press('up')
                time.sleep(self.key_delay)
        
        # Movimiento horizontal
        if col_diff > 0:
            # Mover hacia la derecha
            for _ in range(abs(col_diff)):
                pyautogui.press('right')
                time.sleep(self.key_delay)
        elif col_diff < 0:
            # Mover hacia la izquierda
            for _ in range(abs(col_diff)):
                pyautogui.press('left')
                time.sleep(self.key_delay)
        
        # Actualizar posición
        self.cursor_position = (target_row, target_col)
        time.sleep(self.action_delay)

    def select_cookie(self):
        """Selecciona la cookie en la posición actual del cursor."""
        print(f"[KeyboardExecutor] Seleccionando cookie en {self.cursor_position}")
        pyautogui.press('space')  # o 'enter' según el juego
        time.sleep(self.action_delay)

    def execute_swap(self, pos1: Tuple[int, int], pos2: Tuple[int, int]):
        """
        Ejecuta un intercambio entre dos posiciones.
        
        Args:
            pos1: (fila, columna) primera posición
            pos2: (fila, columna) segunda posición
        """
        print(f"\n[KeyboardExecutor] ===== EJECUTANDO SWAP =====")
        print(f"Posicion 1: {pos1}")
        print(f"Posicion 2: {pos2}")
        
        # Mover a primera posición
        self.move_cursor_to(pos1[0], pos1[1])
        
        # Seleccionar primera cookie
        self.select_cookie()
        
        # Mover a segunda posición
        self.move_cursor_to(pos2[0], pos2[1])
        
        # Seleccionar segunda cookie (esto ejecuta el swap)
        self.select_cookie()
        
        print(f"[KeyboardExecutor] ===== SWAP COMPLETADO =====\n")
        time.sleep(self.action_delay)

    def execute_directional_swap(self, pos: Tuple[int, int], direction: str):
        """
        Ejecuta un swap direccional (movimiento simplificado).
        Útil si el juego permite intercambiar con la tecla direccional.
        
        Args:
            pos: (fila, columna) posición base
            direction: 'up', 'down', 'left', 'right'
        """
        print(f"[KeyboardExecutor] Swap direccional desde {pos} hacia {direction}")
        
        # Mover a la posición
        self.move_cursor_to(pos[0], pos[1])
        
        # Seleccionar
        self.select_cookie()
        
        # Presionar dirección
        pyautogui.press(direction)
        time.sleep(self.action_delay)

    def calculate_direction(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> str:
        """
        Calcula la dirección del movimiento entre dos posiciones adyacentes.
        
        Returns:
            'up', 'down', 'left', 'right'
        """
        row_diff = pos2[0] - pos1[0]
        col_diff = pos2[1] - pos1[1]
        
        if row_diff == -1:
            return 'up'
        elif row_diff == 1:
            return 'down'
        elif col_diff == -1:
            return 'left'
        elif col_diff == 1:
            return 'right'
        else:
            raise ValueError(f"Posiciones no adyacentes: {pos1} -> {pos2}")


class MouseExecutor:
    """Ejecuta movimientos usando clicks del mouse."""
    
    def __init__(self, game_area_screen_coords: Tuple[int, int, int, int],
                 grid_dimensions: Tuple[int, int]):
        """
        Args:
            game_area_screen_coords: (x, y, width, height) del área de juego en coordenadas de pantalla
            grid_dimensions: (rows, cols) dimensiones de la grilla
        """
        self.game_x = game_area_screen_coords[0]
        self.game_y = game_area_screen_coords[1]
        self.game_width = game_area_screen_coords[2]
        self.game_height = game_area_screen_coords[3]
        self.grid_rows = grid_dimensions[0]
        self.grid_cols = grid_dimensions[1]
        
        # Calcular tamaño de celda
        self.cell_width = self.game_width / self.grid_cols
        self.cell_height = self.game_height / self.grid_rows
        
        print(f"[MouseExecutor] Inicializado")
        print(f"  Area de juego: ({self.game_x}, {self.game_y}, {self.game_width}, {self.game_height})")
        print(f"  Grilla: {self.grid_rows}x{self.grid_cols}")
        print(f"  Tamaño de celda: {self.cell_width:.1f}x{self.cell_height:.1f}")

    def grid_to_screen_coords(self, row: int, col: int) -> Tuple[int, int]:
        """
        Convierte coordenadas de grilla a coordenadas de pantalla.
        
        Args:
            row: Fila en la grilla (0-indexed)
            col: Columna en la grilla (0-indexed)
            
        Returns:
            (x, y) coordenadas de pantalla (centro de la celda)
        """
        # Calcular centro de la celda
        x = self.game_x + (col * self.cell_width) + (self.cell_width / 2)
        y = self.game_y + (row * self.cell_height) + (self.cell_height / 2)
        
        return (int(x), int(y))

    def click_position(self, row: int, col: int):
        """Hace click en una posición de la grilla."""
        x, y = self.grid_to_screen_coords(row, col)
        print(f"[MouseExecutor] Click en grilla ({row},{col}) -> pantalla ({x},{y})")
        pyautogui.click(x, y)
        time.sleep(0.1)

    def execute_swap(self, pos1: Tuple[int, int], pos2: Tuple[int, int]):
        """
        Ejecuta un intercambio entre dos posiciones usando clicks.
        
        Args:
            pos1: (fila, columna) primera posición
            pos2: (fila, columna) segunda posición
        """
        print(f"\n[MouseExecutor] ===== EJECUTANDO SWAP =====")
        print(f"Posicion 1: {pos1}")
        print(f"Posicion 2: {pos2}")
        
        # Click en primera posición
        self.click_position(pos1[0], pos1[1])
        
        # Click en segunda posición
        self.click_position(pos2[0], pos2[1])
        
        print(f"[MouseExecutor] ===== SWAP COMPLETADO =====\n")
        time.sleep(0.2)


class MoveExecutor:
    """Clase unificada para ejecutar movimientos."""
    
    def __init__(self, method: str = "keyboard", **kwargs):
        """
        Args:
            method: "keyboard" o "mouse"
            **kwargs: Argumentos adicionales para el executor específico
        """
        self.method = method
        
        if method == "keyboard":
            input_method = kwargs.get('input_method', InputMethod.KEYBOARD_ARROWS)
            self.executor = KeyboardExecutor(input_method)
        elif method == "mouse":
            if 'game_area_coords' not in kwargs or 'grid_dimensions' not in kwargs:
                raise ValueError("MouseExecutor requiere 'game_area_coords' y 'grid_dimensions'")
            self.executor = MouseExecutor(
                kwargs['game_area_coords'],
                kwargs['grid_dimensions']
            )
        else:
            raise ValueError(f"Metodo invalido: {method}. Use 'keyboard' o 'mouse'")
        
        print(f"[MoveExecutor] Inicializado con metodo: {method}")

    def execute_move(self, pos1: Tuple[int, int], pos2: Tuple[int, int], 
                     move_type: str = None):
        """
        Ejecuta un movimiento.
        
        Args:
            pos1: (fila, columna) primera posición
            pos2: (fila, columna) segunda posición
            move_type: 'horizontal' o 'vertical' (opcional)
        """
        self.executor.execute_swap(pos1, pos2)

    def reset(self):
        """Resetea el estado del executor."""
        if hasattr(self.executor, 'reset_cursor_position'):
            self.executor.reset_cursor_position()


def test_executor():
    """Función de prueba."""
    print("\n" + "="*50)
    print("TEST: Move Executor")
    print("="*50)
    
    print("\n¿Que metodo deseas probar?")
    print("1. Teclado (flechas)")
    print("2. Mouse (clicks)")
    
    opcion = input("Selecciona (1 o 2): ").strip()
    
    if opcion == "1":
        # Test con teclado
        executor = MoveExecutor(method="keyboard")
        
        print("\nPRUEBA: Mover cursor y hacer swap")
        print("Posiciona la ventana del juego y presiona ENTER...")
        input()
        
        # Simular un movimiento simple
        print("\nMoviendo de (0,0) a (1,1)...")
        executor.execute_move((0, 0), (0, 1))
        
    elif opcion == "2":
        # Test con mouse
        print("\nConfigura el area de juego:")
        x = int(input("X: "))
        y = int(input("Y: "))
        width = int(input("Ancho: "))
        height = int(input("Alto: "))
        
        print("\nDimensiones de la grilla:")
        rows = int(input("Filas: "))
        cols = int(input("Columnas: "))
        
        executor = MoveExecutor(
            method="mouse",
            game_area_coords=(x, y, width, height),
            grid_dimensions=(rows, cols)
        )
        
        print("\nPosiciona la ventana del juego y presiona ENTER...")
        input()
        
        # Simular click
        print("\nHaciendo click en (0,0) y (0,1)...")
        executor.execute_move((0, 0), (0, 1))
    
    else:
        print("Opcion invalida")
    
    print("\n[OK] Test completado")


if __name__ == "__main__":
    test_executor()
