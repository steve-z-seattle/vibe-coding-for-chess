"""AI for chess game using Minimax with Alpha-Beta pruning."""
import copy
from typing import Optional, Tuple, List
from chess_game import ChessGame
from models import Position, Piece, ValidMove

# Piece values
PIECE_VALUES = {
    'pawn': 100,
    'knight': 320,
    'bishop': 330,
    'rook': 500,
    'queen': 900,
    'king': 20000
}

# Position tables (simplified from standard tables)
PAWN_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

KNIGHT_TABLE = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
]

BISHOP_TABLE = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
]

ROOK_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [0, 0, 0, 5, 5, 0, 0, 0]
]

QUEEN_TABLE = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20]
]

KING_MIDDLE_TABLE = [
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [20, 30, 10, 0, 0, 10, 30, 20]
]

POSITION_TABLES = {
    'pawn': PAWN_TABLE,
    'knight': KNIGHT_TABLE,
    'bishop': BISHOP_TABLE,
    'rook': ROOK_TABLE,
    'queen': QUEEN_TABLE,
    'king': KING_MIDDLE_TABLE
}


class ChessAI:
    """Chess AI using Minimax with Alpha-Beta pruning."""

    def __init__(self, depth: int = 3):
        self.depth = depth
        self.nodes_evaluated = 0

    def get_best_move(self, game: ChessGame) -> Optional[Tuple[int, int, int, int]]:
        """Get the best move for the current player."""
        self.nodes_evaluated = 0
        
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        color = game.current_player
        moves = self._get_all_moves(game, color)
        
        # Sort moves for better pruning (captures first)
        moves.sort(key=lambda m: self._move_priority(game, m), reverse=True)
        
        for from_row, from_col, to_row, to_col, move_info in moves:
            # Make move
            new_game = self._simulate_move(game, from_row, from_col, to_row, to_col, move_info)
            if new_game is None:
                continue
            
            value = self._minimax(new_game, self.depth - 1, alpha, beta, False, color)
            
            if value > best_value:
                best_value = value
                best_move = (from_row, from_col, to_row, to_col)
            
            alpha = max(alpha, best_value)
        
        return best_move

    def _get_all_moves(self, game: ChessGame, color: str) -> List[Tuple[int, int, int, int, ValidMove]]:
        """Get all valid moves for a color."""
        moves = []
        for row in range(8):
            for col in range(8):
                piece = game.board[row][col]
                if piece and piece.color == color:
                    valid_moves = game.get_valid_moves(row, col)
                    for move in valid_moves:
                        moves.append((row, col, move.row, move.col, move))
        return moves

    def _move_priority(self, game: ChessGame, move: Tuple[int, int, int, int, ValidMove]) -> int:
        """Calculate move priority for move ordering."""
        from_row, from_col, to_row, to_col, move_info = move
        target = game.board[to_row][to_col]
        if target:
            attacker = game.board[from_row][from_col]
            return PIECE_VALUES[target.type] - PIECE_VALUES[attacker.type] // 10
        return 0

    def _simulate_move(self, game: ChessGame, from_row: int, from_col: int,
                       to_row: int, to_col: int, move_info: ValidMove) -> Optional[ChessGame]:
        """Simulate a move and return a new game state."""
        new_game = copy.deepcopy(game)
        
        piece = new_game.board[from_row][from_col]
        if not piece:
            return None
        
        # Handle en passant
        if move_info.en_passant:
            captured_row = to_row + 1 if new_game.current_player == 'white' else to_row - 1
            new_game.board[captured_row][to_col] = None
        
        # Handle castling
        if move_info.castling:
            if move_info.castling == 'kingSide':
                new_game.board[from_row][5] = new_game.board[from_row][7]
                new_game.board[from_row][7] = None
            else:
                new_game.board[from_row][3] = new_game.board[from_row][0]
                new_game.board[from_row][0] = None
            new_game.castling_rights[new_game.current_player]['kingSide'] = False
            new_game.castling_rights[new_game.current_player]['queenSide'] = False
        
        # Update castling rights
        if piece.type == 'king':
            new_game.castling_rights[new_game.current_player]['kingSide'] = False
            new_game.castling_rights[new_game.current_player]['queenSide'] = False
        if piece.type == 'rook':
            if from_col == 0:
                new_game.castling_rights[new_game.current_player]['queenSide'] = False
            if from_col == 7:
                new_game.castling_rights[new_game.current_player]['kingSide'] = False
        
        # Move the piece
        captured = new_game.board[to_row][to_col]
        new_game.board[to_row][to_col] = piece
        new_game.board[from_row][from_col] = None
        
        # Handle pawn promotion (auto-promote to queen for AI)
        if piece.type == 'pawn' and (to_row == 0 or to_row == 7):
            new_game.board[to_row][to_col] = Piece(color=piece.color, type='queen')
        
        # Update king position
        if piece.type == 'king':
            new_game.king_positions[piece.color] = Position(row=to_row, col=to_col)
        
        # Update en passant target
        if piece.type == 'pawn' and abs(to_row - from_row) == 2:
            new_game.en_passant_target = Position(row=(from_row + to_row) // 2, col=from_col)
        else:
            new_game.en_passant_target = None
        
        # Switch player
        new_game.current_player = 'black' if new_game.current_player == 'white' else 'white'
        
        return new_game

    def _minimax(self, game: ChessGame, depth: int, alpha: float, beta: float,
                 is_minimizing: bool, maximizing_color: str) -> float:
        """Minimax algorithm with Alpha-Beta pruning."""
        self.nodes_evaluated += 1
        
        if depth == 0:
            return self._evaluate_board(game, maximizing_color)
        
        current_color = game.current_player
        moves = self._get_all_moves(game, current_color)
        
        # Sort moves for better pruning
        moves.sort(key=lambda m: self._move_priority(game, m), reverse=True)
        
        if not moves:
            # Checkmate or stalemate
            if game.is_in_check(current_color):
                if current_color == maximizing_color:
                    return float('-inf')
                else:
                    return float('inf')
            return 0  # Stalemate
        
        if is_minimizing:
            min_eval = float('inf')
            for from_row, from_col, to_row, to_col, move_info in moves:
                new_game = self._simulate_move(game, from_row, from_col, to_row, to_col, move_info)
                if new_game is None:
                    continue
                eval = self._minimax(new_game, depth - 1, alpha, beta, False, maximizing_color)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
        else:
            max_eval = float('-inf')
            for from_row, from_col, to_row, to_col, move_info in moves:
                new_game = self._simulate_move(game, from_row, from_col, to_row, to_col, move_info)
                if new_game is None:
                    continue
                eval = self._minimax(new_game, depth - 1, alpha, beta, True, maximizing_color)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval

    def _evaluate_board(self, game: ChessGame, color: str) -> float:
        """Evaluate the board from the perspective of the given color."""
        score = 0
        
        for row in range(8):
            for col in range(8):
                piece = game.board[row][col]
                if piece:
                    value = PIECE_VALUES[piece.type]
                    
                    # Add position value
                    position_value = 0
                    if piece.type in POSITION_TABLES:
                        table = POSITION_TABLES[piece.type]
                        # Mirror the table for black pieces
                        if piece.color == 'white':
                            position_value = table[row][col]
                        else:
                            position_value = table[7 - row][col]
                    
                    if piece.color == color:
                        score += value + position_value
                    else:
                        score -= value + position_value
        
        # Mobility bonus
        own_moves = len(self._get_all_moves(game, color))
        opponent = 'black' if color == 'white' else 'white'
        opp_moves = len(self._get_all_moves(game, opponent))
        score += (own_moves - opp_moves) * 10
        
        # Check bonus
        if game.is_in_check(opponent):
            score += 50
        
        return score
