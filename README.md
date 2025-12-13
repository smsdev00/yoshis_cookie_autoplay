# Yoshi's Cookie AutoPlayer

Sistema automatizado modular para jugar Yoshi's Cookie usando visión por computadora y análisis de movimientos.

## 📁 Estructura del Proyecto

```
yoshis_cookie_autoplay/
├── config.py                    # Configuración (colores, área de juego)
├── yoshi_cookie_detector.py     # Detección de cookies y construcción de grilla
├── movement_analyzer.py         # Análisis de movimientos óptimos
├── game_controller.py           # Control de captura y gestión del juego
├── keyboard_executor.py         # Ejecución de movimientos (teclado/mouse)
├── auto_player.py              # Sistema principal integrado
├── detect_color.py             # Herramienta para calibrar colores
├── imgs/                       # Imágenes de prueba
├── screenshots/                # Capturas automáticas
└── logs/                       # Logs de sesiones
```

## 🚀 Instalación

### Requisitos

```bash
pip install opencv-python numpy scikit-learn pyautogui
```

### Versiones recomendadas
- Python 3.8+
- OpenCV 4.5+
- scikit-learn 1.0+
- pyautogui 0.9.53+

## 🎮 Uso Rápido

### 1. Calibración de Colores (si es necesario)

```bash
python detect_color.py
```

Ajusta los valores HSV para cada color de cookie y actualiza `config.py`.

### 2. Prueba de Detección

```bash
python yoshi_cookie_detector.py
```

Verifica que detecta correctamente las cookies en una imagen estática.

### 3. AutoPlayer Completo

```bash
python auto_player.py
```

Sigue las instrucciones interactivas:
1. Configura la ventana del juego (clicks en esquinas)
2. Selecciona modo de juego
3. ¡Deja que juegue solo!

## 🔧 Configuración Detallada

### Método 1: Teclado (Recomendado para emuladores)

```python
from auto_player import AutoPlayer

player = AutoPlayer()
player.setup_game_window()
player.controller.setup_move_executor(method="keyboard")
player.play_session(num_moves=10)
```

### Método 2: Mouse (Para ventanas nativas)

```python
player = AutoPlayer()
player.setup_game_window()
# Configurar con dimensiones de la grilla detectada
player.controller.setup_move_executor(
    method="mouse",
    grid_dimensions=(5, 5)  # filas x columnas
)
player.play_session(num_moves=10)
```

## 📊 Módulos Independientes

### GameController - Captura y Control

```python
from game_controller import GameController

controller = GameController()
controller.set_game_window(x=100, y=100, width=800, height=600)

# Capturar screenshot
img = controller.capture_screenshot()

# Configurar executor
controller.setup_move_executor(method="keyboard")

# Ejecutar movimiento
controller.execute_move({
    'pos1': (0, 0),
    'pos2': (0, 1),
    'type': 'horizontal'
})
```

### Detector - Visión por Computadora

```python
from yoshi_cookie_detector import ImprovedCookieDetector
from config import CONF

detector = ImprovedCookieDetector(CONF)

# Detectar cookies
cookies = detector.detectar_cookies('imgs/screenshot.jpg')

# Construir grilla
grilla, info = detector.construir_grilla_inteligente(cookies)

print(grilla)  # Array numpy con la grilla
```

### Analyzer - Inteligencia del Juego

```python
from movement_analyzer import CookieMovementAnalyzer
import numpy as np

analyzer = CookieMovementAnalyzer()

# Grilla de ejemplo
grilla = np.array([
    [2, 1, 2, 1, 3],
    [1, 3, 1, 3, 2],
    [3, 2, 3, 2, 1],
    [1, 3, 2, 1, 3],
    [3, 0, 1, 3, 2]
])

# Analizar mejor movimiento
result = analyzer.analyze_optimal_move(grilla, strategy="balanced")
best_move = result['best_move']

print(f"Mejor movimiento: {best_move.pos1} -> {best_move.pos2}")
print(f"Score: {best_move.score}")
```

### Executor - Ejecución de Movimientos

```python
from keyboard_executor import MoveExecutor

# Modo teclado
executor = MoveExecutor(method="keyboard")
executor.execute_move((0, 0), (0, 1))

# Modo mouse
executor = MoveExecutor(
    method="mouse",
    game_area_coords=(100, 100, 800, 600),
    grid_dimensions=(6, 5)
)
executor.execute_move((2, 3), (3, 3))
```

## 🎯 Estrategias Disponibles

- **balanced**: Balance entre score inmediato y setup futuro
- **aggressive**: Maximiza score inmediato
- **defensive**: Prioriza setup y evita riesgos
- **cascade_focused**: Busca cascadas y combos

```python
player.set_strategy("aggressive")
```

## ⚙️ Configuración Avanzada

### Ajustar config.py

```python
CONF = {
    "cookies_colors": {
        "Verde": {"min": [43, 49, 180], "max": [80, 255, 255]},
        "Rojo": {"min": [129, 92, 137], "max": [179, 255, 255]},
        "Amarillo": {"min": [6, 193, 226], "max": [33, 255, 255]},
    },
    "game_area": {
        "x_min": 690,
        "x_max": 1275,
        "y_min": 340,
        "y_max": 940
    },
    "images_path": "./imgs/"
}
```

### Ajustar delays

```python
player.animation_delay = 1.5  # Segundos después de cada movimiento
player.controller.move_executor.executor.key_delay = 0.1  # Delay entre teclas
```

## 🐛 Troubleshooting

### No detecta cookies correctamente

1. Recalibra colores con `detect_color.py`
2. Ajusta `game_area` en `config.py`
3. Verifica iluminación de la pantalla

### Movimientos no se ejecutan

1. Verifica que la ventana del juego tenga foco
2. Ajusta `key_delay` si las teclas se pierden
3. Prueba método mouse si teclado falla

### Grilla con espacios vacíos incorrectos

1. El sistema filtra automáticamente cookies cayendo
2. Verifica que `CLUSTER_TOLERANCE` sea apropiado (35 por defecto)
3. Revisa los logs para ver qué cookies se excluyen

### Failsafe activado

Si el mouse va a la esquina superior izquierda:
- Es una medida de seguridad
- Reposiciona y continúa
- Desactiva con `pyautogui.FAILSAFE = False` (no recomendado)

## 📈 Logging y Métricas

Los logs se guardan automáticamente en `./logs/`:

```
session_20240111_143022.txt
```

Contiene:
- Timestamp de cada movimiento
- Score total y promedio
- Matches creados
- Errores encontrados

## 🔬 Testing Individual

Cada módulo puede probarse independientemente:

```bash
# Test detector
python yoshi_cookie_detector.py

# Test controller
python game_controller.py

# Test executor
python keyboard_executor.py

# Test analyzer
python movement_analyzer.py
```

## 🚦 Flujo de Ejecución

```
1. AutoPlayer.play_session()
   ↓
2. GameController.capture_screenshot()
   ↓
3. ImprovedCookieDetector.detectar_cookies()
   ↓
4. ImprovedCookieDetector.construir_grilla_inteligente()
   ↓
5. CookieMovementAnalyzer.analyze_optimal_move()
   ↓
6. MoveExecutor.execute_move()
   ↓
7. wait_for_animation()
   ↓
8. Volver al paso 2
```

## 🎨 Personalización

### Crear tu propia estrategia

```python
# En movement_analyzer.py
def _adjust_strategy_weights(self, strategy: str):
    if strategy == "mi_estrategia":
        self.strategy_weights = {
            "immediate_score": 0.5,
            "cascade_potential": 0.3,
            "board_setup": 0.1,
            "risk_mitigation": 0.1,
        }
```

### Añadir nuevo método de input

```python
# En keyboard_executor.py
class InputMethod(Enum):
    MI_METODO = "mi_metodo"

# Implementar lógica en KeyboardExecutor o crear nueva clase
```

## 📝 Notas Importantes

- **Failsafe**: Mueve el mouse a la esquina superior izquierda para detener
- **Delays**: Ajusta según velocidad de tu sistema/emulador
- **Calibración**: Cada setup puede requerir recalibrar colores HSV
- **Área de juego**: Debe incluir SOLO el tablero de cookies

## 🤝 Contribuir

Mejoras sugeridas:
1. Detección de Game Over con OCR
2. Detección de power-ups especiales
3. Estrategias más avanzadas con ML
4. Soporte para más variantes del juego

## 📄 Licencia

Proyecto educativo - Uso libre para aprendizaje

---

¿Preguntas? Revisa los comentarios en cada módulo para más detalles técnicos.
