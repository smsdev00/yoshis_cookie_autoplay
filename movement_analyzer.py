import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import copy

class MoveType(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"

@dataclass
class Move:
    """Representa un movimiento posible en el tablero."""
    pos1: Tuple[int, int]  # Posición de la primera cookie (fila, columna)
    pos2: Tuple[int, int]  # Posición de la segunda cookie
    move_type: MoveType
    score: float = 0.0
    matches_created: List[List[Tuple[int, int]]] = None
    cascade_potential: int = 0
    strategic_value: float = 0.0
    explanation: str = ""

class CookieMovementAnalyzer:
    """
    Analizador de movimientos óptimos para Cookie Run.
    Implementa múltiples estrategias de análisis y optimización.
    """
    
    def __init__(self):
        # Configuración de scoring
        self.MATCH_3_SCORE = 100
        self.MATCH_4_SCORE = 300
        self.MATCH_5_SCORE = 500
        self.L_SHAPE_SCORE = 400
        self.T_SHAPE_SCORE = 400
        self.CASCADE_MULTIPLIER = 1.5
        self.STRATEGIC_BONUS = 50
        
        # Mapeo de valores a nombres
        self.COLOR_NAMES = {0: "Vacío", 1: "Verde", 2: "Rojo", 3: "Amarillo"}
        
        # Configuración de estrategias
        self.strategy_weights = {
            "immediate_score": 0.4,      # Importancia del score inmediato
            "cascade_potential": 0.3,    # Importancia del potencial de cascada
            "board_setup": 0.2,          # Importancia de setup futuro
            "risk_mitigation": 0.1       # Importancia de mitigar riesgos
        }

    def analyze_optimal_move(self, board: np.ndarray, strategy: str = "balanced") -> Dict:
        """
        Analiza el tablero y encuentra el movimiento óptimo.
        
        Args:
            board: Array 2D representando el tablero (0=vacío, 1=verde, 2=rojo, 3=amarillo)
            strategy: Estrategia a usar ("aggressive", "defensive", "balanced", "cascade_focused")
            
        Returns:
            Dict con el mejor movimiento y análisis completo
        """
        print(f"\n🎯 === ANÁLISIS DE MOVIMIENTO ÓPTIMO ===")
        print(f"📋 Estrategia: {strategy}")
        print(f"📐 Tablero shape: {board.shape}")
        
        # Ajustar pesos según estrategia
        self._adjust_strategy_weights(strategy)
        
        # 1. Generar todos los movimientos posibles
        possible_moves = self._generate_all_moves(board)
        print(f"🔍 Movimientos posibles encontrados: {len(possible_moves)}")
        
        if not possible_moves:
            return {"best_move": None, "analysis": "No se encontraron movimientos válidos"}
        
        # 2. Evaluar cada movimiento
        evaluated_moves = []
        for move in possible_moves:
            score_data = self._evaluate_move(board, move)
            move.score = score_data["total_score"]
            move.matches_created = score_data["matches"]
            move.cascade_potential = score_data["cascade_potential"]
            move.strategic_value = score_data["strategic_value"]
            move.explanation = score_data["explanation"]
            evaluated_moves.append(move)
        
        # 3. Ordenar por score total
        evaluated_moves.sort(key=lambda m: m.score, reverse=True)
        
        # 4. Análisis detallado del mejor movimiento
        best_move = evaluated_moves[0]
        analysis = self._create_detailed_analysis(board, best_move, evaluated_moves[:5])
        
        return {
            "best_move": best_move,
            "top_moves": evaluated_moves[:5],
            "analysis": analysis,
            "board_state": self._analyze_board_state(board)
        }

    def _adjust_strategy_weights(self, strategy: str):
        """Ajusta los pesos según la estrategia seleccionada."""
        if strategy == "aggressive":
            self.strategy_weights = {
                "immediate_score": 0.6,
                "cascade_potential": 0.3,
                "board_setup": 0.1,
                "risk_mitigation": 0.0
            }
        elif strategy == "defensive":
            self.strategy_weights = {
                "immediate_score": 0.2,
                "cascade_potential": 0.1,
                "board_setup": 0.4,
                "risk_mitigation": 0.3
            }
        elif strategy == "cascade_focused":
            self.strategy_weights = {
                "immediate_score": 0.2,
                "cascade_potential": 0.6,
                "board_setup": 0.2,
                "risk_mitigation": 0.0
            }
        # "balanced" mantiene los pesos por defecto

    def _generate_all_moves(self, board: np.ndarray) -> List[Move]:
        """Genera todos los movimientos posibles en el tablero."""
        moves = []
        rows, cols = board.shape
        
        # Movimientos horizontales
        for r in range(rows):
            for c in range(cols - 1):
                if board[r, c] != 0 and board[r, c+1] != 0:  # Solo intercambiar cookies no vacías
                    move = Move(
                        pos1=(r, c),
                        pos2=(r, c+1),
                        move_type=MoveType.HORIZONTAL
                    )
                    moves.append(move)
        
        # Movimientos verticales
        for r in range(rows - 1):
            for c in range(cols):
                if board[r, c] != 0 and board[r+1, c] != 0:  # Solo intercambiar cookies no vacías
                    move = Move(
                        pos1=(r, c),
                        pos2=(r+1, c),
                        move_type=MoveType.VERTICAL
                    )
                    moves.append(move)
        
        return moves

    def _evaluate_move(self, board: np.ndarray, move: Move) -> Dict:
        """Evalúa un movimiento específico y calcula su score."""
        # Simular el movimiento
        test_board = board.copy()
        test_board[move.pos1], test_board[move.pos2] = test_board[move.pos2], test_board[move.pos1]
        
        # Detectar matches inmediatos
        matches = self._find_all_matches(test_board)
        
        # Calcular score base
        immediate_score = self._calculate_match_score(matches)
        
        # Calcular potencial de cascada
        cascade_potential = self._calculate_cascade_potential(test_board, matches)
        
        # Calcular valor estratégico
        strategic_value = self._calculate_strategic_value(board, test_board, move)
        
        # Score total ponderado
        total_score = (
            immediate_score * self.strategy_weights["immediate_score"] +
            cascade_potential * self.strategy_weights["cascade_potential"] +
            strategic_value * self.strategy_weights["board_setup"]
        )
        
        # Crear explicación
        explanation = self._create_move_explanation(board, move, matches, immediate_score, cascade_potential, strategic_value)
        
        return {
            "total_score": total_score,
            "immediate_score": immediate_score,
            "cascade_potential": cascade_potential,
            "strategic_value": strategic_value,
            "matches": matches,
            "explanation": explanation
        }

    def _find_all_matches(self, board: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Encuentra todos los matches de 3 o más cookies consecutivas."""
        matches = []
        rows, cols = board.shape
        
        # Matches horizontales
        for r in range(rows):
            current_match = []
            current_color = 0
            
            for c in range(cols):
                if board[r, c] == current_color and current_color != 0:
                    current_match.append((r, c))
                else:
                    if len(current_match) >= 3:
                        matches.append(current_match.copy())
                    current_match = [(r, c)] if board[r, c] != 0 else []
                    current_color = board[r, c]
            
            # Verificar el último match de la fila
            if len(current_match) >= 3:
                matches.append(current_match)
        
        # Matches verticales
        for c in range(cols):
            current_match = []
            current_color = 0
            
            for r in range(rows):
                if board[r, c] == current_color and current_color != 0:
                    current_match.append((r, c))
                else:
                    if len(current_match) >= 3:
                        matches.append(current_match.copy())
                    current_match = [(r, c)] if board[r, c] != 0 else []
                    current_color = board[r, c]
            
            # Verificar el último match de la columna
            if len(current_match) >= 3:
                matches.append(current_match)
        
        # Detectar formas especiales (L, T)
        special_matches = self._find_special_shapes(board)
        matches.extend(special_matches)
        
        return matches

    def _find_special_shapes(self, board: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Detecta formas especiales como L y T."""
        special_matches = []
        rows, cols = board.shape
        
        # Detectar formas L y T con verificación de límites adecuada
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                color = board[r, c]
                if color == 0:
                    continue
                
                # Forma T horizontal (verificar que todos los índices estén en límites)
                if (board[r, c-1] == color and board[r, c+1] == color and
                    board[r-1, c] == color and board[r+1, c] == color):
                    special_matches.append([(r, c-1), (r, c), (r, c+1), (r-1, c), (r+1, c)])
        
        # Detectar formas L con verificación de límites más cuidadosa
        for r in range(rows - 2):
            for c in range(cols - 2):
                color = board[r, c]
                if color == 0:
                    continue
                
                # Forma L (esquina superior izquierda) - verificar límites antes de acceder
                if (c + 2 < cols and r + 2 < rows and
                    board[r, c+1] == color and board[r, c+2] == color and
                    board[r+1, c] == color and board[r+2, c] == color):
                    special_matches.append([(r, c), (r, c+1), (r, c+2), (r+1, c), (r+2, c)])
                
                # Forma L (esquina superior derecha)
                if (c + 2 < cols and r + 2 < rows and
                    board[r, c] == color and board[r, c+1] == color and
                    board[r+1, c+2] == color and board[r+2, c+2] == color):
                    special_matches.append([(r, c), (r, c+1), (r, c+2), (r+1, c+2), (r+2, c+2)])
        
        return special_matches

    def _calculate_match_score(self, matches: List[List[Tuple[int, int]]]) -> float:
        """Calcula el score base de los matches encontrados."""
        total_score = 0
        
        for match in matches:
            match_size = len(match)
            if match_size == 3:
                total_score += self.MATCH_3_SCORE
            elif match_size == 4:
                total_score += self.MATCH_4_SCORE
            elif match_size >= 5:
                total_score += self.MATCH_5_SCORE
            
            # Bonus por formas especiales
            if match_size > 4:
                total_score += self.L_SHAPE_SCORE
        
        return total_score

    def _calculate_cascade_potential(self, board: np.ndarray, matches: List[List[Tuple[int, int]]]) -> float:
        """Calcula el potencial de cascada después del movimiento."""
        if not matches:
            return 0
        
        # Simular eliminación de matches
        test_board = board.copy()
        for match in matches:
            for r, c in match:
                test_board[r, c] = 0
        
        # Simular caída de cookies
        test_board = self._simulate_gravity(test_board)
        
        # Buscar nuevos matches después de la caída
        new_matches = self._find_all_matches(test_board)
        
        cascade_score = len(new_matches) * self.CASCADE_MULTIPLIER * self.MATCH_3_SCORE
        
        return cascade_score

    def _simulate_gravity(self, board: np.ndarray) -> np.ndarray:
        """Simula la caída de cookies por gravedad."""
        result = board.copy()
        rows, cols = result.shape
        
        for c in range(cols):
            # Extraer cookies no vacías de la columna
            column = result[:, c]
            non_zero = column[column != 0]
            
            # Rellenar columna: vacíos arriba, cookies abajo
            result[:, c] = 0
            if len(non_zero) > 0:
                result[rows-len(non_zero):, c] = non_zero
        
        return result

    def _calculate_strategic_value(self, original_board: np.ndarray, new_board: np.ndarray, move: Move) -> float:
        """Calcula el valor estratégico del movimiento."""
        strategic_score = 0
        
        # Bonus por crear setups para futuros matches
        potential_setups = self._count_near_matches(new_board)
        strategic_score += potential_setups * self.STRATEGIC_BONUS
        
        # Penalty por crear tablero desequilibrado
        color_distribution = self._analyze_color_distribution(new_board)
        if color_distribution["balance_score"] < 0.3:
            strategic_score -= 50
        
        # Bonus por limpiar áreas problemáticas
        if self._clears_problematic_area(original_board, new_board):
            strategic_score += 100
        
        return strategic_score

    def _count_near_matches(self, board: np.ndarray) -> int:
        """Cuenta posiciones que están a un movimiento de crear un match."""
        near_matches = 0
        rows, cols = board.shape
        
        # Buscar patrones de 2 cookies del mismo color que podrían formar match
        for r in range(rows):
            for c in range(cols - 2):
                # Patrón AA_A o A_AA con verificación de límites
                if board[r, c] != 0:
                    if board[r, c] == board[r, c+1] and c+2 < cols:
                        if board[r, c+2] == 0 or board[r, c+2] == board[r, c]:
                            near_matches += 1
                    if c+2 < cols and board[r, c] == board[r, c+2] and board[r, c+1] == 0:
                        near_matches += 1
        
        # Similar para patrones verticales
        for c in range(cols):
            for r in range(rows - 2):
                if board[r, c] != 0:
                    if board[r, c] == board[r+1, c] and r+2 < rows:
                        if board[r+2, c] == 0 or board[r+2, c] == board[r, c]:
                            near_matches += 1
                    if r+2 < rows and board[r, c] == board[r+2, c] and board[r+1, c] == 0:
                        near_matches += 1
        
        return near_matches

    def _analyze_color_distribution(self, board: np.ndarray) -> Dict:
        """Analiza la distribución de colores en el tablero."""
        total_cookies = np.count_nonzero(board)
        if total_cookies == 0:
            return {"balance_score": 0, "color_counts": {1: 0, 2: 0, 3: 0}, "total_cookies": 0}
        
        color_counts = {}
        for color in [1, 2, 3]:  # Verde, Rojo, Amarillo
            color_counts[color] = np.count_nonzero(board == color)
        
        # Calcular balance (más equilibrado = mejor)
        counts = list(color_counts.values())
        if max(counts) == 0:
            balance_score = 0
        else:
            balance_score = min(counts) / max(counts)
        
        return {
            "color_counts": color_counts,
            "balance_score": balance_score,
            "total_cookies": total_cookies
        }

    def _clears_problematic_area(self, old_board: np.ndarray, new_board: np.ndarray) -> bool:
        """Verifica si el movimiento limpia áreas problemáticas."""
        # Áreas problemáticas: muchas cookies del mismo color agrupadas sin formar match
        old_clusters = self._find_large_clusters(old_board)
        new_clusters = self._find_large_clusters(new_board)
        
        return len(old_clusters) > len(new_clusters)

    def _find_large_clusters(self, board: np.ndarray) -> List:
        """Encuentra clusters grandes de cookies del mismo color."""
        # Implementación simplificada
        clusters = []
        rows, cols = board.shape
        
        for color in [1, 2, 3]:
            color_positions = list(zip(*np.where(board == color)))
            if len(color_positions) > 6:  # Cluster grande
                clusters.append(color_positions)
        
        return clusters

    def _create_move_explanation(self, board: np.ndarray, move: Move, matches: List, 
                                immediate_score: float, cascade_potential: float, strategic_value: float) -> str:
        """Crea una explicación detallada del movimiento."""
        # Obtener colores reales de las posiciones
        pos1_color = self.COLOR_NAMES.get(board[move.pos1], "Desconocido")
        pos2_color = self.COLOR_NAMES.get(board[move.pos2], "Desconocido")
        
        explanation = f"Intercambiar {move.move_type.value} en ({move.pos1[0]},{move.pos1[1]}) ↔ ({move.pos2[0]},{move.pos2[1]})\n"
        explanation += f"Colores: {pos1_color} ↔ {pos2_color}\n"
        
        if matches:
            explanation += f"• Crea {len(matches)} match(es) inmediato(s) (+{immediate_score:.0f} pts)\n"
        
        if cascade_potential > 0:
            explanation += f"• Potencial de cascada (+{cascade_potential:.0f} pts)\n"
        
        if strategic_value != 0:
            explanation += f"• Valor estratégico ({strategic_value:+.0f} pts)\n"
        
        return explanation

    def _create_detailed_analysis(self, board: np.ndarray, best_move: Move, top_moves: List[Move]) -> str:
        """Crea un análisis detallado de la situación del tablero."""
        analysis = f"🏆 MEJOR MOVIMIENTO (Score: {best_move.score:.1f})\n"
        analysis += f"{best_move.explanation}\n"
        
        analysis += f"\n📊 TOP 5 ALTERNATIVAS:\n"
        for i, move in enumerate(top_moves[1:], 2):
            if i <= 5:  # Limitar a top 5
                analysis += f"{i}. Score {move.score:.1f}: {move.move_type.value} ({move.pos1[0]},{move.pos1[1]}) ↔ ({move.pos2[0]},{move.pos2[1]})\n"
        
        board_analysis = self._analyze_board_state(board)
        analysis += f"\n📋 ESTADO DEL TABLERO:\n{board_analysis}\n"
        
        return analysis

    def _analyze_board_state(self, board: np.ndarray) -> str:
        """Analiza el estado general del tablero."""
        color_dist = self._analyze_color_distribution(board)
        near_matches = self._count_near_matches(board)
        
        state = f"• Total cookies: {color_dist['total_cookies']}\n"
        state += f"• Distribución: Verde={color_dist['color_counts'].get(1, 0)}, "
        state += f"Rojo={color_dist['color_counts'].get(2, 0)}, "
        state += f"Amarillo={color_dist['color_counts'].get(3, 0)}\n"
        state += f"• Balance de colores: {color_dist['balance_score']:.2f}\n"
        state += f"• Setups potenciales: {near_matches}\n"
        
        return state

def demo_usage():
    """Ejemplo de uso del analizador."""
    # Ejemplo de tablero (6x5)
    sample_board = np.array([
        [2, 1, 3, 1, 2],
        [1, 2, 1, 3, 1],
        [3, 1, 2, 1, 3],
        [1, 3, 1, 2, 1],
        [2, 1, 3, 1, 2],
        [1, 2, 1, 3, 1]
    ])
    
    print("Tablero inicial:")
    print(sample_board)
    print("\nLeyenda: 0=Vacío, 1=Verde, 2=Rojo, 3=Amarillo")
    
    analyzer = CookieMovementAnalyzer()
    result = analyzer.analyze_optimal_move(sample_board, strategy="balanced")
    
    print("\n=== RESULTADO DEL ANÁLISIS ===")
    print(result["analysis"])
    
    best_move = result["best_move"]
    if best_move:
        print(f"\n🎯 Ejecutar: {best_move.move_type.value} swap entre {best_move.pos1} y {best_move.pos2}")
        print(f"💯 Score esperado: {best_move.score:.1f}")

if __name__ == "__main__":
    demo_usage()