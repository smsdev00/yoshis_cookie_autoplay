from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import CONF


@dataclass(eq=False)
class Cookie:
    """Una pieza detectada, representada una sola vez por su centro."""

    color: str
    x: int
    y: int
    row: int = -1
    col: int = -1
    confidence: float = 1.0


class ImprovedCookieDetector:
    """Detector determinista para las capturas pixel-art de Yoshi's Cookie."""

    COLOR_MAP = {
        "Verde": 1,
        "Rojo": 2,
        "Amarillo": 3,
        "Cuadrada": 4,
        "Yoshi": 5,
    }
    DRAW_COLORS = {
        "Verde": (0, 255, 0),
        "Rojo": (0, 0, 255),
        "Amarillo": (0, 255, 255),
        "Cuadrada": (80, 120, 190),
        "Yoshi": (255, 255, 255),
    }

    def __init__(self, config: dict):
        self.config = config
        self.cookies_colors = config["cookies_colors"]
        self.game_area = config["game_area"]
        self.images_path = config["images_path"]
        self._mira_actual: Optional[Tuple[int, int, int, int]] = None

        detection = config.get("detection", {})
        self.merge_distance = float(detection.get("merge_distance", 30))
        self.neighbor_distance = float(detection.get("neighbor_distance", 108))
        self.axis_tolerance = float(detection.get("axis_tolerance", 28))

    def detectar_cookies(self, imagen_path: str) -> List[Cookie]:
        """Detecta instancias, no contornos de color independientes."""
        image = cv2.imread(imagen_path)
        if image is None:
            raise ValueError(f"No se pudo cargar {imagen_path}")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        candidates: List[Cookie] = []

        # Los símbolos centrales son anclajes estables y no se tocan entre piezas.
        candidates.extend(self._symbol_candidates(hsv, "Rojo", 350, 800, (25, 45), (18, 32)))
        candidates.extend(self._symbol_candidates(hsv, "Verde", 700, 1450, (30, 50), (25, 44)))
        candidates.extend(self._symbol_candidates(hsv, "Amarillo", 750, 1450, (30, 50), (27, 44)))

        # Las piezas cuadradas comparten un bloque marrón grande y uniforme.
        candidates.extend(
            self._mask_candidates(
                hsv,
                lower=(3, 180, 65),
                upper=(12, 255, 130),
                color="Cuadrada",
                area_range=(1700, 2100),
                width_range=(50, 78),
                height_range=(48, 72),
                use_bbox_center=True,
            )
        )

        # El crosshair tapa el símbolo y deja una pequeña marca blanca. No es un
        # tipo de cookie: clasificamos la pieza ocluida con los píxeles que aún
        # sobreviven alrededor del cursor.
        candidates.extend(self._occluded_candidates(hsv))

        cookies = self._merge_candidates(candidates)
        cookies.sort(key=lambda c: (c.y, c.x))
        print(f"[INFO] Instancias detectadas: {len(cookies)}")
        return cookies

    def _symbol_candidates(
        self,
        hsv: np.ndarray,
        color: str,
        min_area: float,
        max_area: float,
        width_range: Tuple[int, int],
        height_range: Tuple[int, int],
    ) -> List[Cookie]:
        ranges = self.cookies_colors[color]
        return self._mask_candidates(
            hsv,
            ranges["min"],
            ranges["max"],
            color,
            (min_area, max_area),
            width_range,
            height_range,
        )

    def _mask_candidates(
        self,
        hsv: np.ndarray,
        lower: Sequence[int],
        upper: Sequence[int],
        color: str,
        area_range: Tuple[float, float],
        width_range: Tuple[int, int],
        height_range: Tuple[int, int],
        use_bbox_center: bool = False,
    ) -> List[Cookie]:
        mask = cv2.inRange(hsv, np.asarray(lower, np.uint8), np.asarray(upper, np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, width, height = cv2.boundingRect(contour)
            if not (area_range[0] <= area <= area_range[1]):
                continue
            if not (width_range[0] <= width <= width_range[1]):
                continue
            if not (height_range[0] <= height <= height_range[1]):
                continue
            if use_bbox_center:
                cx, cy = x + width // 2, y + height // 2
            else:
                cx, cy = self._contour_center(contour)
            if self._in_game_area(cx, cy):
                result.append(Cookie(color, cx, cy, confidence=0.98))
        return result

    def _occluded_candidates(self, hsv: np.ndarray) -> List[Cookie]:
        white = cv2.inRange(hsv, np.array((0, 0, 240), np.uint8), np.array((179, 70, 255), np.uint8))
        yellow_cfg = self.cookies_colors["Amarillo"]
        yellow = cv2.inRange(
            hsv,
            np.asarray(yellow_cfg["min"], np.uint8),
            np.asarray(yellow_cfg["max"], np.uint8),
        )
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, width, height = cv2.boundingRect(contour)
            if not (40 <= area <= 100 and 10 <= width <= 20 and 9 <= height <= 18):
                continue
            cx, cy = x + width // 2, y + height // 2
            if not self._in_game_area(cx, cy):
                continue
            y1, y2 = max(0, cy - 35), min(yellow.shape[0], cy + 36)
            x1, x2 = max(0, cx - 35), min(yellow.shape[1], cx + 36)
            if cv2.countNonZero(yellow[y1:y2, x1:x2]) < 150:
                continue
            # El centro blanco está unos píxeles a la izquierda/arriba del centro
            # geométrico de la pieza en las capturas originales.
            cookie_x, cookie_y = cx + 7, cy + 6
            color = self._classify_under_crosshair(hsv, cookie_x, cookie_y)
            result.append(Cookie(color, cookie_x, cookie_y, confidence=0.96))
        return result

    def _classify_under_crosshair(self, hsv: np.ndarray, x: int, y: int) -> str:
        """Clasifica usando el símbolo/borde que queda visible bajo el cursor."""
        crop = hsv[max(0, y - 30):y + 31, max(0, x - 34):x + 35]

        def pixels(lower: Sequence[int], upper: Sequence[int]) -> int:
            mask = cv2.inRange(
                crop,
                np.asarray(lower, np.uint8),
                np.asarray(upper, np.uint8),
            )
            return cv2.countNonZero(mask)

        brown = pixels((3, 180, 65), (12, 255, 130))
        red_cfg = self.cookies_colors["Rojo"]
        green_cfg = self.cookies_colors["Verde"]
        red = pixels(red_cfg["min"], red_cfg["max"])
        green = pixels(green_cfg["min"], green_cfg["max"])

        # La cuadrada conserva un bloque marrón mucho mayor (>1400 px en las
        # muestras); las redondas verdes tienen alrededor de 380 px marrones.
        if brown > 800:
            return "Cuadrada"
        if red > 80:
            return "Rojo"
        if green > 50:
            return "Verde"
        return "Amarillo"

    def _merge_candidates(self, candidates: List[Cookie]) -> List[Cookie]:
        """Fusiona anclajes coincidentes, priorizando las clases más específicas."""
        priority = {"Yoshi": 5, "Cuadrada": 4, "Rojo": 3, "Verde": 3, "Amarillo": 2}
        merged: List[Cookie] = []
        for candidate in sorted(candidates, key=lambda c: priority[c.color], reverse=True):
            existing = next(
                (
                    item
                    for item in merged
                    if np.hypot(item.x - candidate.x, item.y - candidate.y) <= self.merge_distance
                ),
                None,
            )
            if existing is None:
                merged.append(candidate)
            elif priority[candidate.color] > priority[existing.color]:
                existing.color = candidate.color
                existing.x, existing.y = candidate.x, candidate.y
                existing.confidence = candidate.confidence
        return merged

    def construir_grilla_inteligente(self, cookies: List[Cookie]) -> Tuple[np.ndarray, Dict]:
        """Selecciona el componente inferior izquierdo y lo cuantiza a una grilla."""
        for cookie in cookies:
            cookie.row = cookie.col = -1
        if not cookies:
            return np.empty((0, 0), dtype=int), {}

        components = self._spatial_components(cookies)
        playable = self._choose_bottom_left_component(components)
        excluded = [cookie for cookie in cookies if cookie not in playable]

        row_centers = self._cluster_axis([cookie.y for cookie in playable])
        col_centers = self._cluster_axis([cookie.x for cookie in playable])
        grid = np.zeros((len(row_centers), len(col_centers)), dtype=int)
        cells: Dict[Tuple[int, int], List[Cookie]] = {}

        for cookie in playable:
            row = int(np.argmin(np.abs(np.asarray(row_centers) - cookie.y)))
            col = int(np.argmin(np.abs(np.asarray(col_centers) - cookie.x)))
            if abs(row_centers[row] - cookie.y) > self.axis_tolerance:
                excluded.append(cookie)
                continue
            if abs(col_centers[col] - cookie.x) > self.axis_tolerance:
                excluded.append(cookie)
                continue
            cookie.row, cookie.col = row, col
            cells.setdefault((row, col), []).append(cookie)

        collisions = 0
        for (row, col), items in cells.items():
            if len(items) > 1:
                collisions += len(items) - 1
            best = max(items, key=lambda item: item.confidence)
            grid[row, col] = self.COLOR_MAP[best.color]

        occupied = int(np.count_nonzero(grid))
        confidence = occupied / max(len(playable), 1)
        if collisions:
            confidence *= 0.5
        info = {
            "num_filas": len(row_centers),
            "num_columnas": len(col_centers),
            "cookies_validas": occupied,
            "cookies_excluidas": len(excluded),
            "cookies_excluidas_lista": excluded,
            "filas_centroids": row_centers,
            "columnas_centroids": col_centers,
            "cookies_por_celda": cells,
            "componentes": len(components),
            "colisiones": collisions,
            "confianza": confidence,
        }
        print(
            f"[INFO] Componente jugable inferior izquierdo: {occupied} piezas, "
            f"grilla {grid.shape[0]}x{grid.shape[1]}, confianza {confidence:.2f}"
        )
        return grid, info

    def _spatial_components(self, cookies: List[Cookie]) -> List[List[Cookie]]:
        remaining = set(cookies)
        components = []
        while remaining:
            seed = remaining.pop()
            component = [seed]
            pending = [seed]
            while pending:
                current = pending.pop()
                neighbors = [
                    other
                    for other in remaining
                    if np.hypot(current.x - other.x, current.y - other.y) <= self.neighbor_distance
                ]
                for neighbor in neighbors:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    pending.append(neighbor)
            components.append(component)
        return components

    @staticmethod
    def _choose_bottom_left_component(components: List[List[Cookie]]) -> List[Cookie]:
        # Tamaño evita que un píxel/indicador aislado situado más abajo gane. La
        # posición de la base domina y, a igual base, gana el inicio más izquierdo.
        viable = [component for component in components if len(component) >= 2]
        if not viable:
            viable = components
        global_bottom = max(max(cookie.y for cookie in component) for component in viable)
        bottom_band = [
            component
            for component in viable
            if max(cookie.y for cookie in component) >= global_bottom - 45
        ]
        return min(
            bottom_band,
            key=lambda component: (
                min(cookie.x for cookie in component),
                -len(component),
                -max(cookie.y for cookie in component),
            ),
        )

    def _cluster_axis(self, values: List[int]) -> List[float]:
        clusters: List[List[int]] = []
        for value in sorted(values):
            if not clusters or value - float(np.mean(clusters[-1])) > self.axis_tolerance:
                clusters.append([value])
            else:
                clusters[-1].append(value)
        return [float(np.mean(cluster)) for cluster in clusters]

    @staticmethod
    def _contour_center(contour) -> Tuple[int, int]:
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            x, y, width, height = cv2.boundingRect(contour)
            return x + width // 2, y + height // 2
        return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])

    def _in_game_area(self, x: int, y: int) -> bool:
        area = self.game_area
        return area["x_min"] <= x <= area["x_max"] and area["y_min"] <= y <= area["y_max"]

    def procesar_imagen(
        self,
        imagen_path: str,
        visualize: bool = True,
        output_path: Optional[str] = None,
    ) -> Optional[Dict]:
        try:
            cookies = self.detectar_cookies(imagen_path)
            grid, info = self.construir_grilla_inteligente(cookies)
            if grid.size == 0:
                print(f"[WARN] No se pudo construir una grilla para {imagen_path}")
                return None
            self._print_summary(imagen_path, grid, info)
            if visualize or output_path:
                image = cv2.imread(imagen_path)
                overlay = self._render_results(image, cookies, info)
                if output_path:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(output_path, overlay)
                if visualize:
                    cv2.imshow("Deteccion Yoshi's Cookie", overlay)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            return {"cookies": cookies, "grilla": grid, "info": info}
        except Exception as exc:
            print(f"[ERROR] {imagen_path}: {exc}")
            return None

    def _render_results(self, image: np.ndarray, cookies: List[Cookie], info: Dict) -> np.ndarray:
        result = image.copy()
        area = self.game_area
        cv2.rectangle(
            result,
            (area["x_min"], area["y_min"]),
            (area["x_max"], area["y_max"]),
            self.config["game_area_border"]["color"],
            self.config["game_area_border"]["thickness"],
        )
        excluded = set(info.get("cookies_excluidas_lista", []))
        for cookie in cookies:
            if cookie in excluded or cookie.row < 0:
                cv2.circle(result, (cookie.x, cookie.y), 12, (0, 0, 255), 2)
                continue
            color = self.DRAW_COLORS[cookie.color]
            cv2.circle(result, (cookie.x, cookie.y), 10, color, -1)
            cv2.circle(result, (cookie.x, cookie.y), 12, (0, 0, 0), 2)
            cv2.putText(
                result,
                f"{cookie.row},{cookie.col}:{cookie.color[0]}",
                (cookie.x - 24, cookie.y - 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return result

    @staticmethod
    def _print_summary(path: str, grid: np.ndarray, info: Dict) -> None:
        letters = {0: ".", 1: "V", 2: "R", 3: "A", 4: "C", 5: "Y"}
        print(f"\n{Path(path).name}: {grid.shape[0]}x{grid.shape[1]}")
        for row in grid:
            print(" ".join(letters[int(value)] for value in row))
        print(
            f"Ocupacion: {np.count_nonzero(grid)}/{grid.size}; "
            f"excluidas: {info['cookies_excluidas']}; componentes: {info['componentes']}\n"
        )

    def return_array_of_images_from_folder(self) -> List[str]:
        extensions = {".png", ".jpg", ".jpeg"}
        return sorted(
            str(path)
            for path in Path(self.images_path).iterdir()
            if path.suffix.lower() in extensions
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector de tableros de Yoshi's Cookie")
    parser.add_argument("images", nargs="*", help="Imágenes; por defecto procesa imgs/")
    parser.add_argument("--headless", action="store_true", help="No abrir ventanas OpenCV")
    parser.add_argument("--output-dir", help="Guardar overlays de diagnóstico en este directorio")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detector = ImprovedCookieDetector(CONF)
    images = args.images or detector.return_array_of_images_from_folder()
    failures = 0
    for image_path in images:
        output_path = None
        if args.output_dir:
            output_path = str(Path(args.output_dir) / Path(image_path).name)
        result = detector.procesar_imagen(
            image_path,
            visualize=not args.headless,
            output_path=output_path,
        )
        failures += result is None
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
