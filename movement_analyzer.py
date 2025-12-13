
import numpy as np
from collections import namedtuple

Move = namedtuple('Move', ['pos1', 'pos2', 'type', 'score', 'details'])

class CookieMovementAnalyzer:
    def __init__(self):
        self.strategy_weights = {}
        self.set_strategy("balanced")

    def set_strategy(self, strategy: str):
        """Define los pesos para diferentes estrategias de puntuación."""
        if strategy == "aggressive":
            self.strategy_weights = {
                "immediate_match": 1.0,
                "setup_3": 0.2,
                "setup_4": 0.4,
                "cascade_potential": 0.3,
            }
        elif strategy == "defensive":
            self.strategy_weights = {
                "immediate_match": 0.6,
                "setup_3": 0.5,
                "setup_4": 0.7,
                "cascade_potential": 0.5,
            }
        elif strategy == "cascade_focused":
            self.strategy_weights = {
                "immediate_match": 0.7,
                "setup_3": 0.4,
                "setup_4": 0.6,
                "cascade_potential": 1.0,
            }
        else:  # balanced
            self.strategy_weights = {
                "immediate_match": 0.8,
                "setup_3": 0.3,
                "setup_4": 0.5,
                "cascade_potential": 0.6,
            }

    def analyze_optimal_move(self, grid: np.ndarray, strategy: str = "balanced"):
        """
        Analiza todos los movimientos posibles y devuelve el mejor según la estrategia.
        """
        self.set_strategy(strategy)
        possible_moves = self._find_possible_moves(grid)
        
        if not possible_moves:
            return {"best_move": None, "all_moves": []}

        best_move = max(possible_moves, key=lambda m: m.score)
        
        return {
            "best_move": best_move,
            "all_moves": sorted(possible_moves, key=lambda m: m.score, reverse=True)
        }

    def _find_possible_moves(self, grid: np.ndarray):
        """Encuentra y puntúa todos los movimientos horizontales y verticales."""
        moves = []
        rows, cols = grid.shape
        
        # Movimientos horizontales (swap de columnas)
        for r in range(rows):
            for c in range(cols - 1):
                if grid[r, c] != grid[r, c+1]: # No tiene sentido swapear iguales
                    swapped_grid = self._simulate_move(grid, (r, c), (r, c+1), 'horizontal')
                    score, details = self._evaluate_grid_state(grid, swapped_grid)
                    moves.append(Move((r, c), (r, c+1), 'horizontal', score, details))

        # Movimientos verticales (swap de filas)
        for c in range(cols):
            for r in range(rows - 1):
                if grid[r, c] != grid[r+1, c]:
                    swapped_grid = self._simulate_move(grid, (r, c), (r+1, c), 'vertical')
                    score, details = self._evaluate_grid_state(grid, swapped_grid)
                    moves.append(Move((r, c), (r+1, c), 'vertical', score, details))
        
        return moves

    def _simulate_move(self, grid: np.ndarray, pos1, pos2, move_type: str):
        """Simula un movimiento y la caída de las galletas."""
        temp_grid = grid.copy()
        
        if move_type == 'horizontal':
            # Swap de columnas completas
            temp_grid[:, [pos1[1], pos2[1]]] = temp_grid[:, [pos2[1], pos1[1]]]
        elif move_type == 'vertical':
            # Swap de filas completas
            temp_grid[[pos1[0], pos2[0]], :] = temp_grid[[pos2[0], pos1[0]], :]
            
        return temp_grid

    def _evaluate_grid_state(self, original_grid, new_grid):
        """Calcula el puntaje de un estado del tablero después de un movimiento."""
        score = 0
        details = {}

        # 1. Puntuación por matches inmediatos
        matches = self._find_matches(new_grid)
        immediate_score = len(matches) * 10
        score += immediate_score * self.strategy_weights.get("immediate_match", 1.0)
        details["immediate_matches"] = len(matches)

        # 2. Puntuación por "setups" (dejar 3 o 4 galletas casi en línea)
        setup_score_3, setup_score_4 = self._calculate_setup_score(new_grid)
        score += setup_score_3 * self.strategy_weights.get("setup_3", 0.3)
        score += setup_score_4 * self.strategy_weights.get("setup_4", 0.5)
        details["setups_3"] = setup_score_3
        details["setups_4"] = setup_score_4

        # 3. Puntuación por potencial de cascada (muy simplificado)
        # Un proxy simple es ver si un match deja otros colores iguales adyacentes
        cascade_score = self._calculate_cascade_potential(new_grid, matches)
        score += cascade_score * self.strategy_weights.get("cascade_potential", 0.6)
        details["cascade_potential"] = cascade_score
        
        return score, details

    def _find_matches(self, grid: np.ndarray):
        """Encuentra todas las filas y columnas completas de un solo color."""
        matches = []
        rows, cols = grid.shape
        
        # Check filas
        for r in range(rows):
            if np.all(grid[r, :] == grid[r, 0]) and grid[r, 0] != 0:
                matches.append(f"Fila {r} de color {grid[r, 0]}")
        
        # Check columnas
        for c in range(cols):
            if np.all(grid[:, c] == grid[0, c]) and grid[0, c] != 0:
                matches.append(f"Columna {c} de color {grid[0, c]}")
                
        return matches

    def _calculate_setup_score(self, grid):
        """Puntúa configuraciones que están cerca de formar una línea."""
        rows, cols = grid.shape
        score_3 = 0
        score_4 = 0
        
        # Filas
        for r in range(rows):
            for c in range(cols - 3):
                line = grid[r, c:c+4]
                unique, counts = np.unique(line[line != 0], return_counts=True)
                if len(unique) == 2 and 3 in counts:
                    score_3 += 1
                if len(unique) == 1 and len(line[line != 0]) == 4: # 4 de 5
                    score_4 +=1

        # Columnas
        for c in range(cols):
            for r in range(rows - 3):
                line = grid[r:r+4, c]
                unique, counts = np.unique(line[line != 0], return_counts=True)
                if len(unique) == 2 and 3 in counts:
                    score_3 += 1
                if len(unique) == 1 and len(line[line != 0]) == 4:
                    score_4 +=1
        
        # Considerar líneas de 4 en grillas de 5
        # (simplificado, se puede expandir)
        
        return score_3, score_4

    def _calculate_cascade_potential(self, grid, matches):
        """Estima el potencial de cascada de forma simple."""
        # Esta es una heurística muy básica.
        # Un método mejor analizaría la gravedad y qué fichas caerían dónde.
        if not matches:
            return 0
        
        # Si un match ocurre, las fichas de arriba caen.
        # Si las fichas que caen son del mismo color que las de abajo, es bueno.
        score = 0
        temp_grid = grid.copy()
        
        # Marcar matches como vacíos (simula limpieza)
        # Esto es una simplificación, no simula la caída real
        for match_str in matches:
            parts = match_str.split()
            idx = int(parts[1])
            if parts[0] == "Fila":
                temp_grid[idx, :] = 0
            else:
                temp_grid[:, idx] = 0

        # Ahora, buscar adyacencias verticales del mismo color
        rows, cols = temp_grid.shape
        for c in range(cols):
            for r in range(rows - 1):
                if temp_grid[r, c] != 0 and temp_grid[r, c] == temp_grid[r+1, c]:
                    score += 1 # Potencial de match vertical post-caída
        
        return score


if __name__ == '__main__':
    analyzer = CookieMovementAnalyzer()

    # Grilla de ejemplo del README
    example_grid = np.array([
        [2, 1, 2, 1, 3],
        [1, 3, 1, 3, 2],
        [3, 2, 3, 2, 1],
        [1, 3, 2, 1, 3],
        [3, 0, 1, 3, 2]  # Suponiendo 0 como vacío
    ])
    
    print("Grilla de ejemplo:")
    print(example_grid)
    
    # Analizar con la estrategia balanceada
    result = analyzer.analyze_optimal_move(example_grid, strategy="balanced")
    best_move = result['best_move']

    if best_move:
        print(f"\nMejor movimiento encontrado (balanced):")
        print(f"  Tipo: {best_move.type}")
        print(f"  Mover: {best_move.pos1} <-> {best_move.pos2}")
        print(f"  Puntaje: {best_move.score:.2f}")
        print(f"  Detalles: {best_move.details}")
        
        # Probar otra estrategia
        result_agg = analyzer.analyze_optimal_move(example_grid, strategy="aggressive")
        best_move_agg = result_agg['best_move']
        print(f"\nMejor movimiento encontrado (aggressive):")
        print(f"  Tipo: {best_move_agg.type}")
        print(f"  Mover: {best_move_agg.pos1} <-> {best_move_agg.pos2}")
        print(f"  Puntaje: {best_move_agg.score:.2f}")
        print(f"  Detalles: {best_move_agg.details}")

    else:
        print("\nNo se encontraron movimientos posibles.")

    # Ejemplo de un movimiento que crea una línea
    grid_easy = np.array([
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 2], # Mover la fila 2 a la 1 haría match de 1s
        [2, 3, 4, 5, 1],
        [3, 4, 5, 1, 2],
        [4, 5, 1, 2, 3],
    ])
    
    print("\nGrilla con movimiento obvio:")
    print(grid_easy)
    result_easy = analyzer.analyze_optimal_move(grid_easy)
    best_move_easy = result_easy['best_move']
    
    if best_move_easy:
        print(f"\nMejor movimiento para la grilla fácil:")
        print(f"  Tipo: {best_move_easy.type}")
        print(f"  Mover: {best_move_easy.pos1} <-> {best_move_easy.pos2}")
        print(f"  Puntaje: {best_move_easy.score:.2f}")
    else:
        print("No se encontró movimiento.")
