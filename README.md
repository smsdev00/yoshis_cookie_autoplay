# Yoshi's Cookie AutoPlayer

Autoplayer experimental para Yoshi's Cookie de SNES en Debian KDE/Wayland. El
modo principal usa el framebuffer BMP nativo de **bsnes**: no captura el
escritorio y funciona igual en ventana o pantalla completa.

```text
F12 → BMP 256×224 → detector → array → solver → ydotool → A+dirección → verificar
```

## Configuración utilizada

Los valores predeterminados corresponden a esta instalación:

```text
bsnes:       /home/sms/Documents/bsnes-nightly/bsnes
ROM:         /home/sms/Downloads/Yoshi's Cookie (USA).zip
screenshots: /home/sms/Downloads/Yoshi's Cookie (USA)-*.bmp
fullscreen:  F11
screenshot:  F12
Start:       Keypad8
```

Bindings del Controller Port 1 observados en bsnes:

| Botón SNES | Tecla |
|---|---:|
| Up / Down / Left / Right | W / S / A / D |
| A | O |
| B | I |
| Start | Keypad8 |

La mecánica correcta es mantener el botón **A del SNES** (`O` en esta
configuración) y pulsar una dirección. La tecla `L` está asignada al botón R del
SNES y no sirve para desplazar cookies.

## Dependencias

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo apt install ydotool
```

El modo bsnes usa `python-evdev` y mantiene un teclado virtual abierto durante
toda la sesión para que Plasma Wayland pueda registrarlo. El usuario debe tener
acceso de escritura limitado a `/dev/uinput`; no configures permisos globales.
No ejecutes todo el bot como root. La ROM y bsnes permanecen fuera del
repositorio. `ydotool` y `ydotoold` sólo se conservan para el pipeline histórico.

## Uso seguro por etapas

Puedes inspeccionar el último BMP existente sin `ydotool` y sin enviar ninguna
tecla:

```bash
venv/bin/python -m autoplay.kiosk inspect
```

Con bsnes abierto, la ROM cargada y la ventana enfocada, observa dos capturas
estables y muestra la propuesta. Este modo pulsa F12, pero no toca el juego:

```bash
venv/bin/python -m autoplay.kiosk observe
```

Después prueba exactamente un movimiento:

```bash
venv/bin/python -m autoplay.kiosk single-step --yes-really-execute
```

Para iniciar bsnes, cargar la ROM, activar fullscreen, pulsar Start y jugar sin
límite:

```bash
venv/bin/python -m autoplay.kiosk kiosk --launch --yes-really-execute
```

El lanzamiento muestra una cuenta regresiva de 20 segundos y activa fullscreen.
Después espera 8 segundos y pulsa Keypad8, espera 10 segundos y vuelve a pulsarlo
para llegar a `STAGE START — PUSH START`, espera otros 10 segundos y pulsa
Keypad8 por tercera vez para comenzar. Los tiempos pueden ajustarse con
`--launch-delay`, `--select-stage-delay`, `--level-start-delay` y
`--gameplay-start-delay`.

El lanzador equivalente intenta además inhibir suspensión/idle mediante
`systemd-inhibit` mientras el proceso está activo:

```bash
./run-bsnes-kiosk.sh
```

El script activa el HUD apenas lanza el kiosco, antes de recorrer los menús e
iniciar la partida, para que esté visible desde el comienzo de la grabación.
Se dibuja sobre las franjas negras laterales y muestra estado del loop, stage
estimado a partir de las transiciones confirmadas, movimiento actual, array del
tablero y score táctico del solver. La derecha conserva el progreso histórico y
los últimos errores registrados. Al entrar bsnes en fullscreen, el HUD se
recrea por encima de la ventana del juego para que el compositor no lo oculte.

Las estadísticas se guardan atómicamente en `runtime/stats.json`. Incluyen
sesiones, movimientos totales, máximo stage, corrida más larga, errores y máximo
score táctico. Este último pertenece al solver: no es el score numérico mostrado
por el juego. Para ejecutar sin interfaz lateral, usa directamente el módulo sin
`--hud`:

```bash
venv/bin/python -m autoplay.kiosk kiosk --launch --yes-really-execute
```

Para una prueba limitada:

```bash
venv/bin/python -m autoplay.kiosk kiosk \
  --launch \
  --max-moves 20 \
  --yes-really-execute
```

Se detiene con `Ctrl+C` o creando el archivo:

```bash
mkdir -p runtime
touch runtime/STOP
```

El modo kiosco intenta recuperarse de título/Game Over pulsando Start después de
cinco observaciones fallidas. No borra los BMP de `Downloads`.

## Frecuencia y estabilidad

- Captura por F12 sólo cuando necesita observar el estado.
- Dos arrays idénticos separados 250 ms se consideran estables.
- Después de mover espera al menos 800 ms.
- Cada movimiento se comprueba con una nueva captura.
- Una captura debe medir exactamente 256×224.

El detector ignora las filas que todavía caen desde arriba y las columnas que
entran por la derecha; sólo toma el rectángulo compacto inferior izquierdo.
Si aparece una cookie que aún no sabe clasificar, el kiosco se detiene antes de
mover y guarda el framebuffer en `runtime/unknown-cookies/` para diagnosticarla.

## Limitaciones actuales

- El cursor se detecta visualmente incluso durante su parpadeo. La navegación se
  hace una tecla por vez y cada nueva posición se confirma con una captura antes
  de ejecutar `A+dirección`.
- La pantalla `STAGE START / PUSH START` se reconoce y continúa con un Keypad8.
  La recuperación genérica de título/Game Over aún debe validarse antes de dejar
  el kiosco desatendido.
- No es un bloqueo de pantalla de KDE: es un modo kiosco a pantalla completa.

## Detector histórico de capturas de escritorio

El pipeline anterior sigue disponible:

```bash
./run.sh
./run.sh imgs/R01S01.png
venv/bin/python -m autoplay.cli observe --region 100,80,1250,950
```

## Pruebas

```bash
venv/bin/python -m unittest discover -v
```

Arquitectura:

```text
autoplay/bsnes.py         proceso, F12/BMP y detector nativo
autoplay/kiosk.py         observe/single-step/kiosk y recuperación
autoplay/domain.py        reglas cíclicas y tipos
autoplay/solver.py        ranking de movimientos
autoplay/orchestrator.py  estabilidad, ejecución y verificación
autoplay/adapters.py      entrada ydotool y captura Wayland histórica
```

## bsnes invisible con Docker Compose (etapa 1)

La primera etapa del entorno aislado compila la versión fijada de bsnes y la
ejecuta dentro de una pantalla X11 virtual (`Xvfb`). No abre una ventana en el
escritorio y fuerza backends de audio nulos. La ROM nunca entra en la imagen ni
en Git: Compose la monta como un único archivo de sólo lectura.

Requisitos: Docker Engine con el plugin Compose y una copia local legal de la
ROM en formato ZIP. Crea la configuración local y edita `ROM_PATH` con una ruta
absoluta:

```bash
cp .env.example .env
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f bsnes
```

Detén el servicio sin borrar configuración ni capturas con:

```bash
docker compose stop
```

`BSNES_REF` fija el tag o commit que se compila. El valor predeterminado es el
commit `7d5aa1e656b9171524d01b1b22917197d8121cb4`, elegido porque usa GTK3 y se
compila en la CI oficial sobre Ubuntu actual. Los datos de configuración quedan
en el volumen `bsnes-config`, las
capturas de bsnes en `screenshots` y los diagnósticos del proyecto en
`./runtime`. El servicio de esta etapa sólo arranca el emulador: integrar el bot,
la entrada virtual y la base de tableros corresponde a las etapas siguientes.
