"""
Sistema automatizado completo para jugar Yoshi's Cookie.
Integra: captura de pantalla, detección, análisis y ejecución de movimientos.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

# Importar módulos propios
from config import CONF
from game_controller import GameController
from yoshi_cookie_detector import ImprovedCookieDetector
from movement_analyzer import CookieMovementAnalyzer


class AutoPlayer:
    """Sistema automatizado completo para jugar."""
    
    def __init__(self, game_region: Tuple[int, int, int, int] = None):
        """
        Args:
            game_region: (x, y, width, height) de la ventana del juego
        """
        self.controller = GameController(game_region)
        self.detector = ImprovedCookieDetector(CONF)
        self.analyzer = CookieMovementAnalyzer()
        
        # Configuración
        self.strategy = "balanced"  # balanced, aggressive, defensive, cascade_focused
        self.animation_delay = 1.0  # Segundos a esperar después de cada movimiento
        self.max_moves_per_session = 100  # Límite de seguridad
        
        # Métricas
        self.stats = {
            'moves_executed': 0,
            'total_score': 0,
            'start_time': None,
            'errors': 0,
            'matches_created': 0
        }
        
        # Logs
        self.logs_dir = Path("./logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        print("[AutoPlayer] Inicializado")

    def setup_game_window(self):
        """Configuración interactiva de la ventana del juego."""
        from game_controller import GameWindowFinder
        
        print("\n" + "="*50)
        print("CONFIGURACION INICIAL")
        print("="*50)
        
        region = GameWindowFinder.find_window_interactive()
        self.controller.set_game_window(*region)
        
        # Verificar con screenshot
        print("\n[INFO] Verificando configuracion...")
        img = self.controller.capture_screenshot(save=True)
        
        cv2.imshow("Verificacion - Presiona ESC si esta correcto", img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if key == 27:  # ESC
            print("[OK] Configuracion verificada")
            return True
        else:
            print("[WARN] Configuracion cancelada")
            return False

    def capture_and_detect(self) -> Optional[Dict]:
        """
        Captura pantalla, detecta cookies y construye grilla.
        
        Returns:
            Dict con cookies, grilla e info, o None si hay error
        """
        try:
            # Capturar pantalla
            screenshot = self.controller.capture_screenshot(save=False)
            
            # Guardar temporalmente para procesamiento
            temp_path = Path("./temp_screenshot.jpg")
            cv2.imwrite(str(temp_path), screenshot)
            
            # Detectar cookies
            cookies = self.detector.detectar_cookies(str(temp_path))
            
            if not cookies:
                print("[WARN] No se detectaron cookies")
                return None
            
            # Construir grilla
            grilla, info = self.detector.construir_grilla_inteligente(cookies)
            
            if grilla.size == 0:
                print("[WARN] Grilla vacia")
                return None
            
            return {
                'cookies': cookies,
                'grilla': grilla,
                'info': info,
                'screenshot': screenshot
            }
            
        except Exception as e:
            print(f"[ERROR] Error en captura/deteccion: {e}")
            self.stats['errors'] += 1
            return None

    def analyze_best_move(self, grilla: np.ndarray) -> Optional[Dict]:
        """
        Analiza la grilla y encuentra el mejor movimiento.
        
        Returns:
            Dict con información del mejor movimiento, o None
        """
        try:
            result = self.analyzer.analyze_optimal_move(grilla, strategy=self.strategy)
            
            if result['best_move'] is None:
                print("[WARN] No se encontro movimiento valido")
                return None
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Error en analisis: {e}")
            self.stats['errors'] += 1
            return None

    def execute_game_move(self, move_analysis: Dict):
        """
        Ejecuta el movimiento en el juego.
        Por ahora solo registra, implementaremos la ejecucion real despues.
        """
        best_move = move_analysis['best_move']
        
        print("\n" + "="*50)
        print(f"[AutoPlayer] EJECUTANDO MOVIMIENTO #{self.stats['moves_executed'] + 1}")
        print("="*50)
        print(best_move.explanation)
        print(f"Score esperado: {best_move.score:.1f}")
        print("="*50)
        
        # Registrar movimiento
        self.controller.execute_move({
            'pos1': best_move.pos1,
            'pos2': best_move.pos2,
            'type': best_move.move_type.value,
            'score': best_move.score
        })
        
        # Actualizar estadísticas
        self.stats['moves_executed'] += 1
        self.stats['total_score'] += best_move.score
        if best_move.matches_created:
            self.stats['matches_created'] += len(best_move.matches_created)

    def play_one_move(self) -> bool:
        """
        Ejecuta un ciclo completo: captura, detecta, analiza y ejecuta.
        
        Returns:
            True si se ejecutó exitosamente, False si no
        """
        print(f"\n{'='*50}")
        print(f"MOVIMIENTO #{self.stats['moves_executed'] + 1}")
        print(f"{'='*50}")
        
        # 1. Capturar y detectar
        detection_result = self.capture_and_detect()
        if detection_result is None:
            return False
        
        # 2. Analizar mejor movimiento
        move_analysis = self.analyze_best_move(detection_result['grilla'])
        if move_analysis is None:
            return False
        
        # 3. Ejecutar movimiento
        self.execute_game_move(move_analysis)
        
        # 4. Esperar animación
        self.controller.wait_for_animation(self.animation_delay)
        
        return True

    def play_session(self, num_moves: int = None):
        """
        Juega una sesión completa.
        
        Args:
            num_moves: Número de movimientos a ejecutar. None = continuo hasta game over
        """
        print("\n" + "="*50)
        print("INICIANDO SESION DE JUEGO")
        print("="*50)
        print(f"Estrategia: {self.strategy}")
        print(f"Movimientos objetivo: {num_moves if num_moves else 'Continuo'}")
        print(f"Delay entre movimientos: {self.animation_delay}s")
        print("\n[INFO] Presiona Ctrl+C o mueve mouse a esquina superior izquierda para detener")
        print("="*50 + "\n")
        
        self.stats['start_time'] = datetime.now()
        
        try:
            move_count = 0
            max_moves = num_moves if num_moves else self.max_moves_per_session
            
            while move_count < max_moves:
                # Verificar si el juego terminó
                if self.controller.is_game_over():
                    print("\n[INFO] Game Over detectado")
                    break
                
                # Ejecutar un movimiento
                success = self.play_one_move()
                
                if not success:
                    print("[WARN] Movimiento fallido, reintentando...")
                    time.sleep(0.5)
                    continue
                
                move_count += 1
                
                # Pausa entre movimientos
                time.sleep(0.2)
                
        except KeyboardInterrupt:
            print("\n\n[INFO] Sesion interrumpida por el usuario")
        except Exception as e:
            print(f"\n[ERROR] Error inesperado: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.print_session_summary()

    def print_session_summary(self):
        """Imprime resumen de la sesión."""
        if self.stats['start_time']:
            duration = datetime.now() - self.stats['start_time']
            duration_str = str(duration).split('.')[0]  # Quitar microsegundos
        else:
            duration_str = "N/A"
        
        print("\n" + "="*50)
        print("RESUMEN DE SESION")
        print("="*50)
        print(f"Duracion: {duration_str}")
        print(f"Movimientos ejecutados: {self.stats['moves_executed']}")
        print(f"Score total: {self.stats['total_score']:.1f}")
        print(f"Matches creados: {self.stats['matches_created']}")
        print(f"Errores: {self.stats['errors']}")
        
        if self.stats['moves_executed'] > 0:
            avg_score = self.stats['total_score'] / self.stats['moves_executed']
            print(f"Score promedio por movimiento: {avg_score:.1f}")
        
        print("="*50 + "\n")
        
        # Guardar log
        self._save_session_log()

    def _save_session_log(self):
        """Guarda log de la sesión."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"session_{timestamp}.txt"
        
        with open(log_file, 'w') as f:
            f.write("="*50 + "\n")
            f.write("LOG DE SESION - YOSHI'S COOKIE AUTO PLAYER\n")
            f.write("="*50 + "\n\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Estrategia: {self.strategy}\n")
            f.write(f"Movimientos: {self.stats['moves_executed']}\n")
            f.write(f"Score total: {self.stats['total_score']:.1f}\n")
            f.write(f"Matches: {self.stats['matches_created']}\n")
            f.write(f"Errores: {self.stats['errors']}\n\n")
            
            f.write("Historial de movimientos:\n")
            f.write("-"*50 + "\n")
            for i, move_data in enumerate(self.controller.get_move_history(), 1):
                f.write(f"{i}. {move_data['timestamp']}: {move_data['move']}\n")
        
        print(f"[INFO] Log guardado: {log_file}")

    def set_strategy(self, strategy: str):
        """Cambia la estrategia de juego."""
        valid_strategies = ["balanced", "aggressive", "defensive", "cascade_focused"]
        if strategy in valid_strategies:
            self.strategy = strategy
            print(f"[AutoPlayer] Estrategia cambiada a: {strategy}")
        else:
            print(f"[ERROR] Estrategia invalida. Opciones: {valid_strategies}")


def main():
    """Función principal - modo interactivo."""
    print("\n" + "="*50)
    print("YOSHI'S COOKIE - AUTO PLAYER")
    print("="*50)
    
    # Crear auto player
    player = AutoPlayer()
    
    # Configurar ventana del juego
    if not player.setup_game_window():
        print("[ERROR] Configuracion cancelada")
        return
    
    # Menú de opciones
    while True:
        print("\n" + "="*50)
        print("MENU PRINCIPAL")
        print("="*50)
        print("1. Jugar un solo movimiento (test)")
        print("2. Jugar N movimientos")
        print("3. Jugar continuamente")
        print("4. Cambiar estrategia (actual: {})".format(player.strategy))
        print("5. Configurar delay ({:.1f}s)".format(player.animation_delay))
        print("6. Ver estadisticas")
        print("0. Salir")
        print("="*50)
        
        opcion = input("\nSelecciona opcion: ").strip()
        
        if opcion == "1":
            player.play_one_move()
            
        elif opcion == "2":
            try:
                n = int(input("Numero de movimientos: "))
                player.play_session(num_moves=n)
            except ValueError:
                print("[ERROR] Numero invalido")
                
        elif opcion == "3":
            confirm = input("¿Jugar continuamente? (s/n): ").strip().lower()
            if confirm == 's':
                player.play_session(num_moves=None)
                
        elif opcion == "4":
            print("\nEstrategias disponibles:")
            print("1. balanced")
            print("2. aggressive")
            print("3. defensive")
            print("4. cascade_focused")
            estrategia = input("Selecciona: ").strip().lower()
            player.set_strategy(estrategia)
            
        elif opcion == "5":
            try:
                delay = float(input("Nuevo delay (segundos): "))
                player.animation_delay = delay
                print(f"[OK] Delay configurado: {delay}s")
            except ValueError:
                print("[ERROR] Valor invalido")
                
        elif opcion == "6":
            player.print_session_summary()
            
        elif opcion == "0":
            print("\n[INFO] Saliendo...")
            break
        else:
            print("[ERROR] Opcion invalida")


if __name__ == "__main__":
    main()
