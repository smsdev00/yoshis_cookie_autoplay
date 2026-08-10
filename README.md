# Yoshi's Cookie AutoPlayer

Detector de tableros y base de autoplay para las capturas pixel-art de Yoshi's
Cookie. El detector identifica una sola instancia por pieza, reconoce cinco
tipos y selecciona como tablero jugable el componente situado más abajo a la
izquierda.

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Detección

`run.sh` procesa `imgs/` en modo headless por defecto, por lo que también puede
usarse en CI o sin servidor gráfico:

```bash
./run.sh
```

Procesar imágenes concretas:

```bash
./run.sh imgs/R01S01.png imgs/R02S03.png
```

Guardar imágenes con las detecciones superpuestas:

```bash
./run.sh --output-dir detection-output
```

Abrir una ventana interactiva por imagen:

```bash
./run.sh --gui imgs/R02S03.png
```

La leyenda de salida es:

- `V`: verde
- `R`: roja
- `A`: amarilla
- `C`: cuadrada
- `Y`: Yoshi
- `.`: celda vacía

## Tests

```bash
venv/bin/python -m unittest discover -v
```

Las regresiones cubren las seis escenas únicas disponibles. `001.png`,
`002.png` y `003.png` son copias de `R01S01.png`, `R01S02.png` y
`R01S03.png`, respectivamente.

## Criterio del tablero jugable

El pipeline realiza estas etapas:

1. Detecta el símbolo central o anclaje visual de cada pieza.
2. Cuando encuentra el crosshair, clasifica la pieza ocluida mediante los
   píxeles del símbolo y del borde que siguen visibles; el cursor no se trata
   como un tipo de cookie.
3. Fusiona detecciones cercanas para impedir duplicados por color.
4. Construye componentes de piezas vecinas.
5. Conserva el componente cuya base está más abajo y cuyo inicio está más a la
   izquierda.
6. Cuantiza sus centros en filas y columnas y reporta colisiones y confianza.

Los parámetros geométricos están en `config.py`, dentro de `detection`.

## Estructura

```text
main.py                       detector y CLI
config.py                     colores, región y tolerancias
movement_analyzer.py          análisis experimental de movimientos
autoplay/auto_player.py       coordinación del ciclo de juego
autoplay/game_controller.py   captura y ejecución
autoplay/keyboard_executor.py teclado y mouse
autoplay/quick_start.py       menú interactivo
detect_color.py               calibración HSV
tests/test_detection.py       regresiones con imgs/
```

## Estado del autoplay

La detección y sus regresiones están operativas. La ejecución real requiere una
sesión gráfica, una ventana del juego correctamente delimitada y validar la
semántica exacta de movimiento del emulador. El failsafe de PyAutoGUI permanece
habilitado: mover el mouse a la esquina superior izquierda detiene la entrada.
