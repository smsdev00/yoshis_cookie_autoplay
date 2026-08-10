# Estado del modo bsnes/kiosco

Última actualización: 10 de agosto de 2026.

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

## Bloqueo actual

Debian 12 incluye `ydotool 0.1.8`, que crea un dispositivo uinput por cada
invocación. KDE Plasma Wayland no alcanza a registrar ese teclado antes de que
se emitan y terminen los eventos.

El código actual además usa la sintaxis del ydotool moderno. No conviene seguir
parchando esa versión vieja: necesitamos un dispositivo virtual persistente.

## Próximo paso recomendado

Usar `python-evdev` dentro de `venv` y mantener un `evdev.UInput` abierto durante
toda la sesión.

La instalación fue iniciada pero cancelada. Actualmente `evdev` **no está
instalado**. Reanudar con:

```bash
venv/bin/pip install evdev
```

Después:

1. Crear `PersistentUInputBackend`.
2. Declarar capacidades para W/S/A/D, O, Keypad8, F11 y F12.
3. Abrir `/dev/uinput` una sola vez.
4. Esperar aproximadamente un segundo para que Plasma registre el dispositivo.
5. Reemplazar `BsnesController(YdotoolInputBackend)` por el backend persistente.
6. Mantener el backend ydotool histórico sólo para otros entornos.

Preparación temporal de `/dev/uinput` usada durante la depuración:

```bash
pkexec /sbin/modprobe uinput
pkexec /usr/bin/setfacl -m u:sms:rw /dev/uinput
```

Para uso permanente debe crearse una regla udev limitada o añadir el usuario al
grupo apropiado; no debe configurarse `/dev/uinput` como escritura global.

## Secuencia de validación pendiente

No saltar directamente al loop infinito:

1. Lanzar bsnes manualmente y cargar la ROM.
2. Crear el dispositivo uinput persistente.
3. Enviar sólo F12 y comprobar que aparece `-004.bmp` o superior.
4. Ejecutar `observe`: dos capturas, ningún movimiento.
5. Confirmar que el tablero detectado coincide visualmente.
6. Colocar/calibrar el cursor inicial.
7. Ejecutar `single-step` una sola vez.
8. Comparar array esperado y observado.
9. Corregir seguimiento del cursor si una limpieza cambia dimensiones.
10. Ejecutar un loop limitado a 5 movimientos.
11. Probar recuperación de Game Over y menús.
12. Sólo entonces ejecutar `./run-bsnes-kiosk.sh` sin límite.

## Riesgos y trabajo todavía pendiente

- Detectar visualmente la posición inicial del cursor o confirmar con certeza
  dónde comienza cada etapa. Ahora se supone `(0,0)`.
- Ajustar el cursor cuando desaparece la fila/columna donde se encontraba.
- Confirmar la cantidad exacta de pulsaciones Start para título → menú → juego.
- Validar los cinco tipos normales y la cookie Yoshi. Las muestras actuales
  sólo contienen cuatro apariencias claramente observadas.
- Mejorar el solver con simulación posterior a la eliminación y lookahead.
- Diferenciar de forma visual Game Over, título, pausa y transición de stage;
  actualmente la recuperación se basa en fallos consecutivos.
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

