# Yoshi's Cookie AutoPlayer

Autoplayer experimental para Yoshi's Cookie de SNES. Está orientado a Snes9x
GTK en Debian KDE/Wayland, pero el detector y el solver no dependen del
emulador.

El flujo nuevo es:

```text
Snes9x → captura Wayland → detector OpenCV → array → solver → ydotool → verificación
```

## Estado y seguridad

- El detector pasa las regresiones de las capturas incluidas.
- Las reglas desplazan filas/columnas cíclicamente y consideran la cookie Yoshi
  como comodín.
- Se esperan tres tableros iguales, separados 250 ms, antes de decidir.
- Después de una entrada se esperan al menos 800 ms y se vuelve a detectar.
- `observe` jamás envía teclas.
- `single-step` y `auto` exigen `--yes-really-execute` deliberadamente.
- El cursor interno comienza en `(0, 0)`: antes de ejecutar hay que colocar el
  cursor real en la esquina superior izquierda.

El modo automático sigue siendo experimental. Pruébalo primero en una partida
sin valor y conserva a mano una forma de cerrar el proceso.

## Dependencias

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo apt install snes9x ydotool spectacle
```

Snes9x GTK soporta Wayland. `ydotoold` debe estar iniciado y tener acceso a
`/dev/uinput`; consulta la configuración del paquete Debian. No ejecutes todo el
autoplayer como root.

En Snes9x configura el mando 1 con bindings dedicados:

| Botón SNES | Tecla | evdev |
|---|---:|---:|
| Arriba | F5 | 63 |
| Abajo | F6 | 64 |
| Izquierda | F7 | 65 |
| Derecha | F8 | 66 |
| A | F9 | 67 |

El bot mantiene `A` y pulsa una dirección para desplazar la fila o columna que
contiene al cursor, de acuerdo con el manual del juego.

## Calibrar la captura

La región se expresa como `x,y,ancho,alto` en coordenadas globales. Debe tener
aproximadamente el mismo tamaño que las muestras (alrededor de 1250×950), porque
el detector actual usa geometría de pixel-art calibrada a esa escala.

Primero observa sin enviar entrada:

```bash
venv/bin/python -m autoplay.cli observe \
  --region 100,80,1250,950 \
  --capture spectacle
```

En KDE Plasma, Spectacle es el backend recomendado. `--capture auto` prueba
primero `grim` y cambia a Spectacle si el protocolo de captura no está
disponible.

Una vez que el array sea correcto, coloca manualmente el cursor del juego en la
celda superior izquierda y prueba una sola acción:

```bash
venv/bin/python -m autoplay.cli single-step \
  --region 100,80,1250,950 \
  --capture spectacle \
  --yes-really-execute
```

Finalmente, para una sesión limitada:

```bash
venv/bin/python -m autoplay.cli auto \
  --region 100,80,1250,950 \
  --capture spectacle \
  --max-moves 20 \
  --yes-really-execute
```

Puedes reemplazar los códigos con un JSON y `--keys archivo.json`.

## Detector estático

```bash
./run.sh
./run.sh imgs/R01S01.png
./run.sh --output-dir detection-output
```

Leyenda: `V` verde, `R` roja, `A` amarilla, `C` cuadrada, `Y` Yoshi y `.` vacío.

## Pruebas

```bash
venv/bin/python -m unittest discover -v
```

La arquitectura principal está en:

```text
autoplay/domain.py        reglas y tipos puros
autoplay/solver.py        ranking de movimientos
autoplay/adapters.py      grim/Spectacle y ydotool
autoplay/orchestrator.py  estabilidad y verificación
autoplay/cli.py           observe/single-step/auto
main.py                   visión y CLI de imágenes estáticas
```
