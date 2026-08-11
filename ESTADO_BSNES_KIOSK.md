# Estado del modo bsnes/kiosco

Última actualización: 11 de agosto de 2026.

## Objetivo final

Ejecutar Yoshi's Cookie en bsnes a pantalla completa y dejar que el bot juegue
continuamente como una visualización tipo salvapantallas/kiosco:

```text
iniciar bsnes + ROM
→ fullscreen
→ iniciar partida
→ F12
→ esperar BMP completo
→ detectar tablero
→ calcular movimiento
→ mantener A del SNES + dirección
→ verificar el resultado
→ repetir
```

No será un salvapantallas de bloqueo de KDE. Será una aplicación fullscreen con
inhibición de suspensión mientras esté activa.

## Git

- Rama de trabajo: `feature/bsnes-kiosk`.
- Commit publicado de la primera implementación: `70790bc`.
- La rama sigue a `origin/feature/bsnes-kiosk`.
- No hay cambios de código sin confirmar antes de crear este documento.

## Rutas verificadas

```text
bsnes: /home/sms/Documents/bsnes-nightly/bsnes
ROM:   /home/sms/Downloads/Yoshi's Cookie (USA).zip
BMP:   /home/sms/Downloads/Yoshi's Cookie (USA)-NNN.bmp
```

bsnes tiene configurado `Screenshots: (same as loaded game)`, por eso los BMP
aparecen junto a la ROM.

## Bindings reales de bsnes

Configuración observada en la pantalla Input de bsnes:

| Control SNES | Tecla física | Código evdev |
|---|---:|---:|
| Up | W | 17 |
| Down | S | 31 |
| Left | A | 30 |
| Right | D | 32 |
| B | I | 23 |
| A | O | 24 |
| Y | P | 25 |
| X | J | 36 |
| L | K | 37 |
| R | L | 38 |
| Select | Keypad0 | 82 |
| Start | Keypad8 | 72 |
| Fullscreen | F11 | 87 |
| Screenshot | F12 | 88 |

Importante: para mover cookies se mantiene presionado el botón **A del mando
SNES**, que en esta configuración es la tecla `O`. La tecla `L` física está
asignada al botón R del SNES y no mueve cookies.

La mecánica fue contrastada con el manual: una dirección sola mueve el cursor;
A + izquierda/derecha desplaza cíclicamente la fila; A + arriba/abajo desplaza
cíclicamente la columna. El cursor se mueve junto con la cookie seleccionada.

## Lo implementado

### Integración bsnes

- `autoplay/bsnes.py`
  - rutas predeterminadas de bsnes, ROM y screenshots;
  - lanzamiento de bsnes con la ROM;
  - F11, F12, Start y controles configurados;
  - espera de un BMP nuevo y de tamaño estable;
  - detector directo del framebuffer nativo `256×224`;
  - seguimiento del cursor después de una rotación.
- `autoplay/kiosk.py`
  - modos `inspect`, `observe`, `single-step` y `kiosk`;
  - lanzamiento opcional de bsnes;
  - recuperación básica mediante Start tras varios fallos;
  - parada por `Ctrl+C`, SIGTERM o `runtime/STOP`.
- `run-bsnes-kiosk.sh`
  - inicia el kiosco;
  - utiliza `systemd-inhibit` para intentar impedir idle/suspensión.

### Detector nativo

Se verificaron tres capturas reales de bsnes:

```text
Yoshi's Cookie (USA)-001.bmp → tablero 3×3
Yoshi's Cookie (USA)-002.bmp → tablero 4×3
Yoshi's Cookie (USA)-003.bmp → tablero 4×4
```

El detector ignora correctamente la fila que todavía cae desde arriba y la
columna que entra por la derecha. Los arrays obtenidos fueron:

```text
001: [[3,4,3], [1,3,1], [2,1,2]]
002: [[3,2,3], [3,2,3], [1,3,1], [2,1,2]]
003: [[3,2,3,3], [3,4,3,1], [1,3,1,2], [2,1,2,1]]
```

El modo sin entrada funciona:

```bash
venv/bin/python -m autoplay.kiosk inspect
```

En la última prueba propuso desplazar hacia arriba la columna de índice 1.

### Pruebas

La última ejecución completa dio:

```text
Ran 16 tests
OK
```

Incluye pruebas del framebuffer nativo, espera del BMP, desplazamientos
cíclicos, comodín Yoshi, solver, estabilidad y detector histórico.

## Pruebas físicas realizadas

1. Se instaló el paquete Debian `ydotool 0.1.8-3+b2`.
2. Se cargó temporalmente `uinput` y se otorgó una ACL temporal al usuario
   `sms` sobre `/dev/uinput`.
3. bsnes pudo ejecutarse directamente con la ROM.
4. Se intentó `observe --launch`.
5. No apareció un BMP nuevo porque F12 no llegó a bsnes.
6. Se descubrió que el backend implementado usaba la sintaxis del ydotool
   moderno (`88:1 88:0`), mientras Debian 12 usa nombres (`F12`, `o+w`).
7. También se probó la sintaxis antigua correcta con retardo. Plasma Wayland no
   reconoció el teclado virtual efímero y Alt+Tab/F12 tampoco llegaron.
8. No se ejecutó ningún movimiento de cookies.

El proceso bsnes de depuración ya no estaba activo al escribir este documento.
`/dev/uinput` tampoco estaba presente, por lo que el módulo/ACL temporal deberá
prepararse nuevamente.

## Entrada persistente implementada

Debian 12 incluye `ydotool 0.1.8`, que crea un dispositivo uinput por cada
invocación. KDE Plasma Wayland no alcanza a registrar ese teclado antes de que
se emitan y terminen los eventos.

El modo bsnes ahora usa `PersistentUInputBackend`, basado en `python-evdev`:

- declara W/S/A/D, O, Keypad8, F11 y F12;
- abre un único `evdev.UInput` durante toda la sesión;
- espera un segundo para que Plasma registre el dispositivo;
- conserva el orden A presionada, dirección, A liberada;
- cierra el dispositivo al salir, incluso después de un error;
- mantiene ydotool sólo para el pipeline histórico.

`evdev 1.9.3` está instalado en `venv` y forma parte de `requirements.txt`. Las
37 pruebas automatizadas pasan.

La validación física confirmó que el mismo dispositivo persistente envía F11,
Start y F12 de forma repetida a bsnes: se generaron los BMP `004` a `047`. Tras
la espera visible de arranque, fullscreen y el primer Start funcionaron y se
llegó al menú `ACTION / PUSH START`. `observe --launch` ya tolera frames de
menú/transición.

La secuencia de inicio validada usa tres Keypad8 temporizados. El detector ahora
localiza la mira, clasifica la cookie visible bajo ella y conserva el cursor
durante su parpadeo. La navegación es cerrada: después de cada dirección se
captura nuevamente hasta confirmar la celda objetivo.

Se validó un `single-step` completo y un loop limitado a cinco movimientos. Los
cinco fueron verificados y el bot atravesó automáticamente dos pantallas
`STAGE START`, enviando un solo Keypad8 en cada transición.

Las pruebas extendidas produjeron diagnósticos reales de todas las variantes
encontradas. El detector distingue los cinco tipos normales, Yoshi como comodín,
las fases clara/oscura del cursor y cookies ocluidas. Ante cualquier apariencia
desconocida guarda el framebuffer en `runtime/unknown-cookies/` y se detiene.
Los fallos posteriores a enviar un movimiento también son fatales para evitar
continuar desincronizado. Se alcanzaron 16 movimientos en una corrida de
depuración, aunque esa ejecución reveló fallos intermedios y no cuenta como una
validación limpia de 20 movimientos.

## Próximo paso recomendado

Repetir una ejecución limpia limitada a 20 movimientos y después validar Game
Over/título antes de habilitar el loop infinito.

Preparación temporal de `/dev/uinput` usada durante la depuración:

```bash
pkexec /sbin/modprobe uinput
pkexec /usr/bin/setfacl -m u:sms:rw /dev/uinput
```

Para uso permanente debe crearse una regla udev limitada o añadir el usuario al
grupo apropiado; no debe configurarse `/dev/uinput` como escritura global.

## Secuencia de validación

Completado:

1. Lanzamiento de bsnes y carga de ROM.
2. Dispositivo uinput persistente.
3. F12 repetido y BMP completo.
4. `observe` sin movimientos.
5. Detección visual del cursor y cookie ocluida.
6. `single-step` con array esperado/observado idéntico.
7. Loop limitado a cinco movimientos.
8. Transición automática entre stages mediante `STAGE START`.

Pendiente:

1. Ejecutar una prueba limpia limitada a 20 movimientos.
2. Probar Game Over/título sin recuperación ciega.
3. Sólo entonces ejecutar `./run-bsnes-kiosk.sh` sin límite.

## Riesgos y trabajo todavía pendiente

- Ajustar el cursor cuando desaparece la fila/columna donde se encontraba.
- Ampliar regresiones con nuevas fases visuales si aparecen; los cinco tipos
  normales y Yoshi ya fueron observados y clasificados.
- Mejorar el solver con simulación posterior a la eliminación y lookahead.
- Diferenciar visualmente Game Over, título y pausa. La transición normal de
  stage ya se reconoce; la recuperación restante se basa en fallos consecutivos.
- Evitar que `Downloads` acumule BMP indefinidamente, preferiblemente cambiando
  Screenshots en bsnes a una carpeta exclusiva; el bot no debe borrar archivos
  generales automáticamente.
- Crear una inhibición específica del bloqueo de KDE si se desea; `systemd-inhibit`
  no convierte el programa en un lock screen real.

## Comandos útiles

```bash
# Estado
git status --short --branch

# Inspección sin enviar teclas
venv/bin/python -m autoplay.kiosk inspect

# Pruebas
venv/bin/python -m unittest discover -v

# Parar kiosco
mkdir -p runtime
touch runtime/STOP
```
