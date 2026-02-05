"""AI for chess game using Minimax with Alpha-Beta pruning and Transposition Table."""
import time
from typing import Optional, Tuple, List, Dict
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

# Transposition Table Entry Types
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


class TranspositionEntry:
    """Entry in the transposition table."""
    def __init__(self, depth: int, value: float, flag: int, best_move=None):
        self.depth = depth
        self.value = value
        self.flag = flag
        self.best_move = best_move


class ChessAI:
    """Chess AI using Minimax with Alpha-Beta pruning and Transposition Table."""

    def __init__(self, depth: int = 3, max_time: float = 3.0):
        self.max_depth = depth
        self.max_time = max_time  # Maximum thinking time in seconds
        self.nodes_evaluated = 0
        self.transposition_table: Dict[int, TranspositionEntry] = {}
        self.start_time = 0
        self.time_limit_reached = False
        self.killer_moves = [[None, None] for _ in range(20)]  # Killer heuristic

    def get_best_move(self, game: ChessGame) -> Optional[Tuple[int, int, int, int]]:
        """Get the best move for the current player using iterative deepening."""
        self.nodes_evaluated = 0
        self.transposition_table.clear()
        self.start_time = time.time()
        self.time_limit_reached = False
        self.killer_moves = [[None, None] for _ in range(20)]
        
        color = game.current_player
        best_move = None
        
        # Iterative deepening - search progressively deeper until time runs out
        for depth in range(1, self.max_depth + 1):
            if time.time() - self.start_time > self.max_time * 0.8:
                break
                
            try:
                move = self._search_at_depth(game, depth, color)
                if move:
                    best_move = move
            except TimeoutError:
                break
        
        return best_move

    def _search_at_depth(self, game: ChessGame, depth: int, color: str) -> Optional[Tuple[int, int, int, int]]:
        """Search at a specific depth."""
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        moves = self._get_all_moves_ordered(game, color, 0)
        
        for from_row, from_col, to_row, to_col, move_info in moves:
            if self._check_timeout():
                raise TimeoutError()
            
            # Make move incrementally (much faster than deepcopy)
            undo_info = self._make_move_fast(game, from_row, from_col, to_row, to_col, move_info)
            if undo_info is None:
                continue
            
            try:
                value = self._minimax(game, depth - 1, alpha, beta, False, color, 1)
            finally:
                # Always undo the move, even if TimeoutError is raised
                self._undo_move_fast(game, undo_info)
            
            if value > best_value:
                best_value = value
                best_move = (from_row, from_col, to_row, to_col)
            
            alpha = max(alpha, best_value)
        
        return best_move

    def _check_timeout(self) -> bool:
        """Check if time limit is approaching."""
        if time.time() - self.start_time > self.max_time:
            self.time_limit_reached = True
            return True
        return False

    def _get_all_moves_ordered(self, game: ChessGame, color: str, depth: int) -> List:
        """Get all valid moves with better ordering for alpha-beta pruning."""
        moves = []
        for row in range(8):
            for col in range(8):
                piece = game.board[row][col]
                if piece and piece.color == color:
                    valid_moves = game.get_valid_moves(row, col)
                    for move in valid_moves:
                        moves.append((row, col, move.row, move.col, move))
        
        # Enhanced move ordering
        def move_score(m):
            from_row, from_col, to_row, to_col, move_info = m
            score = 0
            
            # 1. TT move (best move from transposition table)
            board_hash = self._hash_board(game)
            if board_hash in self.transposition_table:
                entry = self.transposition_table[board_hash]
                if entry.best_move == m[:4]:
                    score += 100000
            
            # 2. MVV-LVA: Most Valuable Victim - Least Valuable Aggressor
            target = game.board[to_row][to_col]
            if target:
                attacker = game.board[from_row][from_col]
                score += 10000 + PIECE_VALUES[target.type] - PIECE_VALUES[attacker.type] // 10
            
            # 3. Killer heuristic
            if depth < len(self.killer_moves):
                if self.killer_moves[depth][0] == m[:4]:
                    score += 9000
                elif self.killer_moves[depth][1] == m[:4]:
                    score += 8000
            
            # 4. Promotion bonus
            if move_info and hasattr(move_info, 'promotion'):
                score += 5000
            
            return score
        
        moves.sort(key=move_score, reverse=True)
        return moves

    def _hash_board(self, game: ChessGame) -> int:
        """Create a simple hash of the board state."""
        h = 0
        for row in range(8):
            for col in range(8):
                piece = game.board[row][col]
                if piece:
                    h ^= hash((row, col, piece.color, piece.type)) * 31
        h ^= hash((game.current_player, 
                   game.castling_rights['white']['kingSide'],
                   game.castling_rights['white']['queenSide'],
                   game.castling_rights['black']['kingSide'],
                   game.castling_rights['black']['queenSide']))
        return h

    def _make_move_fast(self, game: ChessGame, from_row: int, from_col: int,
                        to_row: int, to_col: int, move_info: ValidMove) -> Optional[dict]:
        """Make a move and return undo information (much faster than deepcopy)."""
        piece = game.board[from_row][from_col]
        if not piece:
            return None
        
        # Save state for undo
        undo_info = {
            'from_row': from_row,
            'from_col': from_col,
            'to_row': to_row,
            'to_col': to_col,
            'piece': piece,
            'captured': game.board[to_row][to_col],
            'en_passant_target': game.en_passant_target,
            'castling_rights': {
                'white': dict(game.castling_rights['white']),
                'black': dict(game.castling_rights['black'])
            },
            'king_positions': {
                'white': Position(row=game.king_positions['white'].row, col=game.king_positions['white'].col),
                'black': Position(row=game.king_positions['black'].row, col=game.king_positions['black'].col)
            },
            'current_player': game.current_player,
            'en_passant_capture': None
        }
        
        # Handle en passant
        if move_info and move_info.en_passant:
            captured_row = to_row + 1 if game.current_player == 'white' else to_row - 1
            undo_info['en_passant_capture'] = (captured_row, to_col, game.board[captured_row][to_col])
            game.board[captured_row][to_col] = None
        
        # Handle castling
        if move_info and move_info.castling:
            if move_info.castling == 'kingSide':
                undo_info['castled_rook_from'] = (from_row, 7)
                undo_info['castled_rook_to'] = (from_row, 5)
                game.board[from_row][5] = game.board[from_row][7]
                game.board[from_row][7] = None
            else:
                undo_info['castled_rook_from'] = (from_row, 0)
                undo_info['castled_rook_to'] = (from_row, 3)
                game.board[from_row][3] = game.board[from_row][0]
                game.board[from_row][0] = None
            game.castling_rights[game.current_player]['kingSide'] = False
            game.castling_rights[game.current_player]['queenSide'] = False
        
        # Update castling rights
        if piece.type == 'king':
            game.castling_rights[game.current_player]['kingSide'] = False
            game.castling_rights[game.current_player]['queenSide'] = False
        if piece.type == 'rook':
            if from_col == 0:
                game.castling_rights[game.current_player]['queenSide'] = False
            if from_col == 7:
                game.castling_rights[game.current_player]['kingSide'] = False
        
        # Move the piece
        game.board[to_row][to_col] = piece
        game.board[from_row][from_col] = None
        
        # Handle pawn promotion (auto-promote to queen for AI)
        if piece.type == 'pawn' and (to_row == 0 or to_row == 7):
            undo_info['promoted'] = True
            game.board[to_row][to_col] = Piece(color=piece.color, type='queen')
        
        # Update king position
        if piece.type == 'king':
            game.king_positions[piece.color] = Position(row=to_row, col=to_col)
        
        # Update en passant target
        if piece.type == 'pawn' and abs(to_row - from_row) == 2:
            game.en_passant_target = Position(row=(from_row + to_row) // 2, col=from_col)
        else:
            game.en_passant_target = None
        
        # Switch player
        game.current_player = 'black' if game.current_player == 'white' else 'white'
        
        return undo_info

    def _undo_move_fast(self, game: ChessGame, undo_info: dict):
        """Undo a move using stored information."""
        # Switch back player
        game.current_player = undo_info['current_player']
        
        from_row, from_col = undo_info['from_row'], undo_info['from_col']
        to_row, to_col = undo_info['to_row'], undo_info['to_col']
        
        # Undo promotion
        if undo_info.get('promoted'):
            game.board[from_row][from_col] = undo_info['piece']
        else:
            game.board[from_row][from_col] = undo_info['piece']
        
        game.board[to_row][to_col] = undo_info['captured']
        
        # Undo en passant capture
        if undo_info['en_passant_capture']:
            captured_row, captured_col, captured_piece = undo_info['en_passant_capture']
            game.board[captured_row][captured_col] = captured_piece
        
        # Undo castling
        if 'castled_rook_from' in undo_info:
            rook_from_row, rook_from_col = undo_info['castled_rook_from']
            rook_to_row, rook_to_col = undo_info['castled_rook_to']
            game.board[rook_from_row][rook_from_col] = game.board[rook_to_row][rook_to_col]
            game.board[rook_to_row][rook_to_col] = None
        
        # Restore state
        game.en_passant_target = undo_info['en_passant_target']
        game.castling_rights = undo_info['castling_rights']
        game.king_positions = undo_info['king_positions']

    def _minimax(self, game: ChessGame, depth: int, alpha: float, beta: float,
                 is_minimizing: bool, maximizing_color: str, current_depth: int) -> float:
        """Minimax algorithm with Alpha-Beta pruning and Transposition Table."""
        self.nodes_evaluated += 1
        
        if self._check_timeout():
            raise TimeoutError()
        
        # Transposition Table Lookup
        board_hash = self._hash_board(game)
        tt_entry = self.transposition_table.get(board_hash)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == EXACT:
                return tt_entry.value
            elif tt_entry.flag == LOWERBOUND and tt_entry.value >= beta:
                return tt_entry.value
            elif tt_entry.flag == UPPERBOUND and tt_entry.value <= alpha:
                return tt_entry.value
        
        if depth == 0:
            return self._evaluate_board(game, maximizing_color)
        
        current_color = game.current_player
        moves = self._get_all_moves_ordered(game, current_color, current_depth)
        
        if not moves:
            # Checkmate or stalemate
            if game.is_in_check(current_color):
                if current_color == maximizing_color:
                    return float('-inf') + current_depth  # Prefer faster checkmate
                else:
                    return float('inf') - current_depth
            return 0  # Stalemate
        
        best_value = float('inf') if is_minimizing else float('-inf')
        best_move = None
        original_alpha = alpha
        original_beta = beta
        
        for from_row, from_col, to_row, to_col, move_info in moves:
            undo_info = self._make_move_fast(game, from_row, from_col, to_row, to_col, move_info)
            if undo_info is None:
                continue
            
            try:
                eval = self._minimax(game, depth - 1, alpha, beta, not is_minimizing, maximizing_color, current_depth + 1)
            finally:
                # Always undo the move, even if TimeoutError is raised
                self._undo_move_fast(game, undo_info)
            
            if is_minimizing:
                if eval < best_value:
                    best_value = eval
                    best_move = (from_row, from_col, to_row, to_col)
                beta = min(beta, eval)
                if beta <= alpha:
                    # Killer move heuristic
                    if current_depth < len(self.killer_moves):
                        if self.killer_moves[current_depth][0] != best_move:
                            self.killer_moves[current_depth][1] = self.killer_moves[current_depth][0]
                            self.killer_moves[current_depth][0] = best_move
                    break
            else:
                if eval > best_value:
                    best_value = eval
                    best_move = (from_row, from_col, to_row, to_col)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    # Killer move heuristic
                    if current_depth < len(self.killer_moves):
                        if self.killer_moves[current_depth][0] != best_move:
                            self.killer_moves[current_depth][1] = self.killer_moves[current_depth][0]
                            self.killer_moves[current_depth][0] = best_move
                    break
        
        # Store in Transposition Table
        if best_value <= original_alpha:
            flag = UPPERBOUND
        elif best_value >= original_beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        
        self.transposition_table[board_hash] = TranspositionEntry(depth, best_value, flag, best_move)
        
        return best_value

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
                        if piece.color == 'white':
                            position_value = table[row][col]
                        else:
                            position_value = table[7 - row][col]
                    
                    if piece.color == color:
                        score += value + position_value
                    else:
                        score -= value + position_value
        
        # Mobility bonus (simplified - count legal moves)
        own_moves = 0
        opp_moves = 0
        opponent = 'black' if color == 'white' else 'white'
        
        for row in range(8):
            for col in range(8):
                piece = game.board[row][col]
                if piece:
                    moves = len(game.get_valid_moves(row, col))
                    if piece.color == color:
                        own_moves += moves
                    else:
                        opp_moves += moves
        
        score += (own_moves - opp_moves) * 5
        
        # Check bonus
        if game.is_in_check('black' if color == 'white' else 'white'):
            score += 50
        
        return score
