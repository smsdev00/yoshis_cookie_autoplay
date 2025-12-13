"""
Script de inicio rápido para el AutoPlayer.
Configuración asistida paso a paso.
"""

import sys
from pathlib import Path


def print_banner():
    """Imprime banner de bienvenida."""
    print("\n" + "="*60)
    print(" "*15 + "YOSHI'S COOKIE AUTOPLAYER")
    print(" "*20 + "Inicio Rapido")
    print("="*60 + "\n")


def check_dependencies():
    """Verifica que todas las dependencias estén instaladas."""
    print("[1/5] Verificando dependencias...")
    
    missing = []
    
    try:
        import cv2
        print("  [OK] OpenCV instalado")
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import numpy
        print("  [OK] NumPy instalado")
    except ImportError:
        missing.append("numpy")
    
    try:
        import sklearn
        print("  [OK] scikit-learn instalado")
    except ImportError:
        missing.append("scikit-learn")
    
    try:
        import pyautogui
        print("  [OK] PyAutoGUI instalado")
    except ImportError:
        missing.append("pyautogui")
    
    if missing:
        print("\n[ERROR] Faltan dependencias:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstala con: pip install " + " ".join(missing))
        return False
    
    print("\n[OK] Todas las dependencias instaladas\n")
    return True


def check_files():
    """Verifica que todos los archivos necesarios existan."""
    print("[2/5] Verificando archivos del proyecto...")
    
    required_files = [
        'config.py',
        'yoshi_cookie_detector.py',
        'movement_analyzer.py',
        'game_controller.py',
        'keyboard_executor.py',
        'auto_player.py'
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"  [OK] {file}")
        else:
            print(f"  [X] {file} - FALTA")
            missing.append(file)
    
    if missing:
        print(f"\n[ERROR] Faltan {len(missing)} archivo(s)")
        return False
    
    print("\n[OK] Todos los archivos presentes\n")
    return True


def setup_directories():
    """Crea directorios necesarios."""
    print("[3/5] Configurando directorios...")
    
    dirs = ['imgs', 'screenshots', 'logs']
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir()
            print(f"  [CREADO] {dir_name}/")
        else:
            print(f"  [OK] {dir_name}/")
    
    print("\n[OK] Directorios configurados\n")
    return True


def choose_mode():
    """Permite elegir el modo de ejecución."""
    print("[4/5] Selecciona modo de ejecucion:")
    print("\n  1. Test de Deteccion (imagen estatica)")
    print("  2. Configurar Ventana del Juego")
    print("  3. Jugar Automaticamente (un movimiento)")
    print("  4. Jugar Sesion Completa")
    print("  5. Calibrar Colores HSV")
    print("  0. Salir")
    
    return input("\nOpcion: ").strip()


def run_detection_test():
    """Ejecuta test de detección."""
    print("\n" + "="*60)
    print("TEST DE DETECCION")
    print("="*60 + "\n")
    
    from yoshi_cookie_detector import ImprovedCookieDetector
    from config import CONF
    
    # Buscar imagen de prueba
    imgs_dir = Path("imgs")
    images = list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.png"))
    
    if not images:
        print("[ERROR] No hay imagenes en imgs/")
        print("Coloca una imagen del juego en imgs/ y reintenta")
        return
    
    print("Imagenes disponibles:")
    for i, img in enumerate(images, 1):
        print(f"  {i}. {img.name}")
    
    try:
        choice = int(input("\nSelecciona imagen: ")) - 1
        img_path = images[choice]
    except (ValueError, IndexError):
        print("[ERROR] Seleccion invalida")
        return
    
    print(f"\nProcesando: {img_path}")
    
    detector = ImprovedCookieDetector(CONF)
    resultado = detector.procesar_imagen(str(img_path))
    
    if resultado:
        print("\n[OK] Test completado exitosamente")
        input("\nPresiona ENTER para continuar...")
    else:
        print("\n[ERROR] Fallo en la deteccion")
        input("\nPresiona ENTER para continuar...")


def run_window_setup():
    """Ejecuta configuración de ventana."""
    print("\n" + "="*60)
    print("CONFIGURACION DE VENTANA")
    print("="*60 + "\n")
    
    print("Instrucciones:")
    print("1. Abre el juego Yoshi's Cookie")
    print("2. Deja la ventana visible")
    print("3. Presiona ENTER para continuar")
    
    input("\nPresiona ENTER...")
    
    from game_controller import GameController, GameWindowFinder
    
    controller = GameController()
    region = GameWindowFinder.find_window_interactive()
    controller.set_game_window(*region)
    
    # Test screenshot
    print("\n[INFO] Capturando screenshot de prueba...")
    img = controller.capture_screenshot(save=True)
    
    import cv2
    cv2.imshow("Screenshot - Presiona ESC si esta correcto", img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if key == 27:  # ESC
        # Guardar configuración
        print("\n[OK] Ventana configurada correctamente")
        print(f"Region: x={region[0]}, y={region[1]}, w={region[2]}, h={region[3]}")
        
        # Guardar en archivo para uso futuro
        config_file = Path("window_config.txt")
        with open(config_file, 'w') as f:
            f.write(f"{region[0]},{region[1]},{region[2]},{region[3]}")
        
        print(f"\n[INFO] Configuracion guardada en {config_file}")
        input("\nPresiona ENTER para continuar...")
    else:
        print("\n[CANCELADO] Intenta de nuevo")
        input("\nPresiona ENTER para continuar...")


def run_single_move():
    """Ejecuta un solo movimiento."""
    print("\n" + "="*60)
    print("PRUEBA DE MOVIMIENTO UNICO")
    print("="*60 + "\n")
    
    # Verificar si existe configuración guardada
    config_file = Path("window_config.txt")
    
    if not config_file.exists():
        print("[WARN] No hay configuracion de ventana guardada")
        print("Ejecuta primero la opcion 2 (Configurar Ventana)")
        input("\nPresiona ENTER para continuar...")
        return
    
    # Cargar configuración
    with open(config_file, 'r') as f:
        x, y, w, h = map(int, f.read().strip().split(','))
    
    print(f"[INFO] Usando configuracion guardada: x={x}, y={y}, w={w}, h={h}")
    
    from auto_player import AutoPlayer
    
    player = AutoPlayer(game_region=(x, y, w, h))
    
    print("\n[INFO] Selecciona metodo de input:")
    print("  1. Teclado (flechas + espacio)")
    print("  2. Mouse (clicks)")
    
    method_choice = input("\nMetodo: ").strip()
    
    if method_choice == "1":
        player.controller.setup_move_executor(method="keyboard")
    elif method_choice == "2":
        print("\n[INFO] Necesito las dimensiones de la grilla")
        print("Ejecuta primero un test de deteccion para saberlas")
        rows = int(input("Filas: "))
        cols = int(input("Columnas: "))
        player.controller.setup_move_executor(
            method="mouse",
            grid_dimensions=(rows, cols)
        )
    else:
        print("[ERROR] Opcion invalida")
        return
    
    print("\n[INFO] Asegurate de que:")
    print("  - La ventana del juego este visible")
    print("  - El juego este listo para jugar")
    print("  - No hay otras ventanas encima")
    
    input("\nPresiona ENTER para ejecutar movimiento...")
    
    success = player.play_one_move()
    
    if success:
        print("\n[OK] Movimiento ejecutado")
    else:
        print("\n[ERROR] Fallo al ejecutar movimiento")
    
    input("\nPresiona ENTER para continuar...")


def run_full_session():
    """Ejecuta sesión completa."""
    print("\n" + "="*60)
    print("SESION COMPLETA")
    print("="*60 + "\n")
    
    config_file = Path("window_config.txt")
    
    if not config_file.exists():
        print("[WARN] No hay configuracion de ventana guardada")
        print("Ejecuta primero la opcion 2 (Configurar Ventana)")
        input("\nPresiona ENTER para continuar...")
        return
    
    with open(config_file, 'r') as f:
        x, y, w, h = map(int, f.read().strip().split(','))
    
    from auto_player import AutoPlayer
    
    player = AutoPlayer(game_region=(x, y, w, h))
    
    # Configurar método
    print("[INFO] Metodo de input:")
    print("  1. Teclado")
    print("  2. Mouse")
    
    method_choice = input("\nMetodo: ").strip()
    
    if method_choice == "1":
        player.controller.setup_move_executor(method="keyboard")
    elif method_choice == "2":
        rows = int(input("Filas de la grilla: "))
        cols = int(input("Columnas de la grilla: "))
        player.controller.setup_move_executor(
            method="mouse",
            grid_dimensions=(rows, cols)
        )
    
    # Configurar estrategia
    print("\n[INFO] Estrategia:")
    print("  1. Balanced")
    print("  2. Aggressive")
    print("  3. Defensive")
    print("  4. Cascade Focused")
    
    strategy_choice = input("\nEstrategia: ").strip()
    strategies = {"1": "balanced", "2": "aggressive", "3": "defensive", "4": "cascade_focused"}
    player.set_strategy(strategies.get(strategy_choice, "balanced"))
    
    # Número de movimientos
    try:
        num_moves = int(input("\nNumero de movimientos (0 = continuo): "))
        if num_moves == 0:
            num_moves = None
    except ValueError:
        num_moves = 10
    
    print("\n" + "="*60)
    print("INICIANDO SESION")
    print("="*60)
    print(f"Movimientos: {num_moves if num_moves else 'Continuo'}")
    print(f"Estrategia: {player.strategy}")
    print("\n[IMPORTANTE] Mueve el mouse a la esquina superior izquierda para detener")
    
    input("\nPresiona ENTER para iniciar...")
    
    player.play_session(num_moves=num_moves)


def run_color_calibration():
    """Ejecuta calibración de colores."""
    print("\n" + "="*60)
    print("CALIBRACION DE COLORES HSV")
    print("="*60 + "\n")
    
    print("Este proceso te ayudara a ajustar los rangos HSV para cada color")
    print("\nInstrucciones:")
    print("1. Usa los sliders para ajustar los rangos")
    print("2. Presiona ESC cuando estes satisfecho")
    print("3. Anota los valores y actualiza config.py")
    
    input("\nPresiona ENTER para continuar...")
    
    try:
        import detect_color
        detect_color.hsv_color_selector()
    except Exception as e:
        print(f"[ERROR] Error en calibracion: {e}")
    
    input("\nPresiona ENTER para continuar...")


def main():
    """Función principal."""
    print_banner()
    
    # Verificaciones iniciales
    if not check_dependencies():
        sys.exit(1)
    
    if not check_files():
        print("\n[ERROR] Proyecto incompleto")
        print("Asegurate de tener todos los archivos en el directorio")
        sys.exit(1)
    
    setup_directories()
    
    # Loop principal
    while True:
        mode = choose_mode()
        
        if mode == "1":
            run_detection_test()
        elif mode == "2":
            run_window_setup()
        elif mode == "3":
            run_single_move()
        elif mode == "4":
            run_full_session()
        elif mode == "5":
            run_color_calibration()
        elif mode == "0":
            print("\n[INFO] Saliendo...")
            break
        else:
            print("\n[ERROR] Opcion invalida")
            input("\nPresiona ENTER para continuar...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrumpido por el usuario")
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {e}")
        import traceback
        traceback.print_exc()
