"""Chess game logic."""
import copy
from typing import List, Optional, Dict, Any, Tuple
from models import Position, Piece, ValidMove, GameState


class ChessGame:
    """A chess game implementation."""

    def __init__(self):
        self.board: List[List[Optional[Piece]]] = []
        self.current_player = 'white'
        self.move_history: List[Dict[str, Any]] = []
        self.captured_by_white: List[Piece] = []
        self.captured_by_black: List[Piece] = []
        self.last_move: Optional[Dict[str, Any]] = None
        self.king_positions = {
            'white': Position(row=7, col=4),
            'black': Position(row=0, col=4)
        }
        self.castling_rights = {
            'white': {'kingSide': True, 'queenSide': True},
            'black': {'kingSide': True, 'queenSide': True}
        }
        self.en_passant_target: Optional[Position] = None
        self.pending_promotion: Optional[Dict[str, Any]] = None
        self.init_board()

    def init_board(self):
        """Initialize the chess board."""
        self.board = [
            [Piece(color='black', type='rook'), Piece(color='black', type='knight'),
             Piece(color='black', type='bishop'), Piece(color='black', type='queen'),
             Piece(color='black', type='king'), Piece(color='black', type='bishop'),
             Piece(color='black', type='knight'), Piece(color='black', type='rook')],
            [Piece(color='black', type='pawn') for _ in range(8)],
            [None] * 8,
            [None] * 8,
            [None] * 8,
            [None] * 8,
            [Piece(color='white', type='pawn') for _ in range(8)],
            [Piece(color='white', type='rook'), Piece(color='white', type='knight'),
             Piece(color='white', type='bishop'), Piece(color='white', type='queen'),
             Piece(color='white', type='king'), Piece(color='white', type='bishop'),
             Piece(color='white', type='knight'), Piece(color='white', type='rook')]
        ]
        self.current_player = 'white'
        self.move_history = []
        self.captured_by_white = []
        self.captured_by_black = []
        self.last_move = None
        self.king_positions = {'white': Position(row=7, col=4), 'black': Position(row=0, col=4)}
        self.castling_rights = {
            'white': {'kingSide': True, 'queenSide': True},
            'black': {'kingSide': True, 'queenSide': True}
        }
        self.en_passant_target = None
        self.pending_promotion = None

    def get_valid_moves(self, row: int, col: int) -> List[ValidMove]:
        """Get valid moves for a piece at the given position."""
        piece = self.board[row][col]
        if not piece:
            return []
        
        moves = self._get_possible_moves(row, col)
        
        # Filter out moves that would put the king in check
        valid_moves = []
        for move in moves:
            # Simulate the move
            original_target = self.board[move.row][move.col]
            original_king_pos = copy.copy(self.king_positions[piece.color])
            
            self.board[move.row][move.col] = piece
            self.board[row][col] = None
            
            # Update king position if king is moving
            if piece.type == 'king':
                self.king_positions[piece.color] = Position(row=move.row, col=move.col)
            
            # Check if the king would be in check
            in_check = self.is_in_check(piece.color)
            
            # Restore the board
            self.board[row][col] = piece
            self.board[move.row][move.col] = original_target
            self.king_positions[piece.color] = original_king_pos
            
            if not in_check:
                valid_moves.append(move)
        
        return valid_moves

    def _get_possible_moves(self, row: int, col: int) -> List[ValidMove]:
        """Get possible moves for a piece (without checking for check)."""
        piece = self.board[row][col]
        if not piece:
            return []
        
        moves = []
        
        if piece.type == 'pawn':
            moves.extend(self._get_pawn_moves(row, col, piece.color))
        elif piece.type == 'knight':
            moves.extend(self._get_knight_moves(row, col, piece.color))
        elif piece.type == 'bishop':
            moves.extend(self._get_bishop_moves(row, col, piece.color))
        elif piece.type == 'rook':
            moves.extend(self._get_rook_moves(row, col, piece.color))
        elif piece.type == 'queen':
            moves.extend(self._get_bishop_moves(row, col, piece.color))
            moves.extend(self._get_rook_moves(row, col, piece.color))
        elif piece.type == 'king':
            moves.extend(self._get_king_moves(row, col, piece.color))
        
        return moves

    def _get_pawn_moves(self, row: int, col: int, color: str) -> List[ValidMove]:
        """Get pawn moves."""
        moves = []
        direction = -1 if color == 'white' else 1
        start_row = 6 if color == 'white' else 1
        
        # Move forward one square
        new_row = row + direction
        if self._is_valid_square(new_row, col) and not self.board[new_row][col]:
            moves.append(ValidMove(row=new_row, col=col))
            
            # Move forward two squares from starting position
            if row == start_row and not self.board[row + 2 * direction][col]:
                moves.append(ValidMove(row=row + 2 * direction, col=col))
        
        # Diagonal captures
        for dc in [-1, 1]:
            new_col = col + dc
            if self._is_valid_square(new_row, new_col):
                target = self.board[new_row][new_col]
                if target and target.color != color:
                    moves.append(ValidMove(row=new_row, col=new_col))
                
                # En passant
                if self.en_passant_target and \
                   self.en_passant_target.row == new_row and \
                   self.en_passant_target.col == new_col:
                    moves.append(ValidMove(row=new_row, col=new_col, en_passant=True))
        
        return moves

    def _get_knight_moves(self, row: int, col: int, color: str) -> List[ValidMove]:
        """Get knight moves."""
        moves = []
        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        
        for dr, dc in deltas:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_square(new_row, new_col):
                target = self.board[new_row][new_col]
                if not target or target.color != color:
                    moves.append(ValidMove(row=new_row, col=new_col))
        
        return moves

    def _get_bishop_moves(self, row: int, col: int, color: str) -> List[ValidMove]:
        """Get bishop moves."""
        return self._get_sliding_moves(row, col, color, [(-1, -1), (-1, 1), (1, -1), (1, 1)])

    def _get_rook_moves(self, row: int, col: int, color: str) -> List[ValidMove]:
        """Get rook moves."""
        return self._get_sliding_moves(row, col, color, [(-1, 0), (1, 0), (0, -1), (0, 1)])

    def _get_sliding_moves(self, row: int, col: int, color: str, directions: List[Tuple[int, int]]) -> List[ValidMove]:
        """Get sliding piece moves (bishop, rook, queen)."""
        moves = []
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            while self._is_valid_square(new_row, new_col):
                target = self.board[new_row][new_col]
                if not target:
                    moves.append(ValidMove(row=new_row, col=new_col))
                else:
                    if target.color != color:
                        moves.append(ValidMove(row=new_row, col=new_col))
                    break
                new_row += dr
                new_col += dc
        
        return moves

    def _get_king_moves(self, row: int, col: int, color: str) -> List[ValidMove]:
        """Get king moves."""
        moves = []
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in deltas:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_square(new_row, new_col):
                target = self.board[new_row][new_col]
                if not target or target.color != color:
                    moves.append(ValidMove(row=new_row, col=new_col))
        
        # Castling
        if not self.is_in_check(color):
            # King side
            if self._can_castle(color, 'kingSide'):
                moves.append(ValidMove(row=row, col=col + 2, castling='kingSide'))
            # Queen side
            if self._can_castle(color, 'queenSide'):
                moves.append(ValidMove(row=row, col=col - 2, castling='queenSide'))
        
        return moves

    def _can_castle(self, color: str, side: str) -> bool:
        """Check if castling is possible."""
        row = 7 if color == 'white' else 0
        rights = self.castling_rights[color][side]
        
        if not rights:
            return False
        
        if side == 'kingSide':
            if self.board[row][5] or self.board[row][6]:
                return False
            if self._is_square_attacked(row, 5, color) or self._is_square_attacked(row, 6, color):
                return False
        else:
            if self.board[row][1] or self.board[row][2] or self.board[row][3]:
                return False
            if self._is_square_attacked(row, 2, color) or self._is_square_attacked(row, 3, color):
                return False
        
        return True

    def _is_valid_square(self, row: int, col: int) -> bool:
        """Check if a square is valid."""
        return 0 <= row < 8 and 0 <= col < 8

    def is_in_check(self, color: str) -> bool:
        """Check if a color is in check."""
        king_pos = self.king_positions[color]
        return self._is_square_attacked(king_pos.row, king_pos.col, color)

    def _is_square_attacked(self, row: int, col: int, defending_color: str) -> bool:
        """Check if a square is attacked by the opponent."""
        attacking_color = 'black' if defending_color == 'white' else 'white'
        
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece.color == attacking_color:
                    # Special handling for pawns (they attack diagonally)
                    if piece.type == 'pawn':
                        direction = -1 if attacking_color == 'white' else 1
                        if r + direction == row and abs(c - col) == 1:
                            return True
                    elif piece.type == 'king':
                        if abs(r - row) <= 1 and abs(c - col) <= 1:
                            return True
                    else:
                        moves = self._get_possible_moves(r, c)
                        if any(m.row == row and m.col == col for m in moves):
                            return True
        
        return False

    def make_move(self, from_row: int, from_col: int, to_row: int, to_col: int,
                  promotion_piece: Optional[str] = None) -> Tuple[bool, str]:
        """Make a move on the board."""
        piece = self.board[from_row][from_col]
        if not piece:
            return False, "No piece at the starting position"
        
        if piece.color != self.current_player:
            return False, "Not your turn"
        
        # Check if the move is valid
        valid_moves = self.get_valid_moves(from_row, from_col)
        move = None
        for m in valid_moves:
            if m.row == to_row and m.col == to_col:
                move = m
                break
        
        if not move:
            return False, "Invalid move"
        
        captured = self.board[to_row][to_col]
        
        # Handle en passant
        if move.en_passant:
            captured_row = to_row + 1 if self.current_player == 'white' else to_row - 1
            captured = self.board[captured_row][to_col]
            self.board[captured_row][to_col] = None
        
        # Record captured piece
        if captured:
            if self.current_player == 'white':
                self.captured_by_white.append(captured)
            else:
                self.captured_by_black.append(captured)
        
        # Handle castling
        if move.castling:
            if move.castling == 'kingSide':
                self.board[from_row][5] = self.board[from_row][7]
                self.board[from_row][7] = None
            else:
                self.board[from_row][3] = self.board[from_row][0]
                self.board[from_row][0] = None
            self.castling_rights[self.current_player]['kingSide'] = False
            self.castling_rights[self.current_player]['queenSide'] = False
        
        # Update castling rights
        if piece.type == 'king':
            self.castling_rights[self.current_player]['kingSide'] = False
            self.castling_rights[self.current_player]['queenSide'] = False
        if piece.type == 'rook':
            if from_col == 0:
                self.castling_rights[self.current_player]['queenSide'] = False
            if from_col == 7:
                self.castling_rights[self.current_player]['kingSide'] = False
        
        # Move the piece
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = None
        
        # Handle promotion
        if promotion_piece:
            self.board[to_row][to_col] = Piece(color=self.current_player, type=promotion_piece)
        
        # Update king position
        if piece.type == 'king':
            self.king_positions[self.current_player] = Position(row=to_row, col=to_col)
        
        # Set en passant target
        if piece.type == 'pawn' and abs(to_row - from_row) == 2:
            self.en_passant_target = Position(row=(from_row + to_row) // 2, col=from_col)
        else:
            self.en_passant_target = None
        
        # Record the move
        self.last_move = {'from': Position(row=from_row, col=from_col),
                          'to': Position(row=to_row, col=to_col)}
        
        # Save state to history
        self.move_history.append({
            'piece': piece,
            'from': Position(row=from_row, col=from_col),
            'to': Position(row=to_row, col=to_col),
            'captured': captured,
            'board': copy.deepcopy(self.board),
            'castling_rights': copy.deepcopy(self.castling_rights),
            'king_positions': copy.deepcopy(self.king_positions),
            'en_passant_target': self.en_passant_target,
            'current_player': self.current_player
        })
        
        # Switch player
        self.current_player = 'black' if self.current_player == 'white' else 'white'
        
        return True, "Move successful"

    def undo_move(self) -> bool:
        """Undo the last move (reverts both players' moves)."""
        if len(self.move_history) < 2:
            return False
        
        # Remove the last two moves from history (one for each player)
        self.move_history.pop()  # Remove opponent's move
        self.move_history.pop()  # Remove current player's move
        
        # Restore to the state before both moves
        if len(self.move_history) > 0:
            # Restore from the last remaining state in history
            last_state = self.move_history[-1]
            self.board = copy.deepcopy(last_state['board'])
            self.castling_rights = copy.deepcopy(last_state['castling_rights'])
            self.king_positions = copy.deepcopy(last_state['king_positions'])
            self.en_passant_target = last_state['en_passant_target']
            # After undo, it's the opponent's turn (they move next)
            self.current_player = 'black' if last_state['current_player'] == 'white' else 'white'
            
            # Recalculate captured pieces from remaining history
            self.captured_by_white = []
            self.captured_by_black = []
            for move in self.move_history:
                if move['captured']:
                    if move['piece'].color == 'white':
                        self.captured_by_white.append(move['captured'])
                    else:
                        self.captured_by_black.append(move['captured'])
            
            # Update last move
            self.last_move = {
                'from': self.move_history[-1]['from'],
                'to': self.move_history[-1]['to']
            }
        else:
            # No moves left, reset to initial state
            self.init_board()
        
        return True

    def has_any_valid_moves(self, color: str) -> bool:
        """Check if a color has any valid moves."""
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece and piece.color == color:
                    moves = self.get_valid_moves(row, col)
                    if moves:
                        return True
        return False

    def is_insufficient_material(self) -> bool:
        """Check for insufficient material draw."""
        pieces = {'white': [], 'black': []}
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece and piece.type != 'king':
                    pieces[piece.color].append({**piece.model_dump(), 'row': row, 'col': col})
        
        # King vs King
        if not pieces['white'] and not pieces['black']:
            return True
        
        # King and minor piece vs King
        if (not pieces['white'] and len(pieces['black']) == 1) or \
           (not pieces['black'] and len(pieces['white']) == 1):
            lone_piece = pieces['white'][0] if pieces['white'] else pieces['black'][0]
            if lone_piece['type'] in ['bishop', 'knight']:
                return True
        
        # Both sides have only a bishop on the same color
        if len(pieces['white']) == 1 and len(pieces['black']) == 1 and \
           pieces['white'][0]['type'] == 'bishop' and pieces['black'][0]['type'] == 'bishop':
            white_square = pieces['white'][0]['row'] + pieces['white'][0]['col']
            black_square = pieces['black'][0]['row'] + pieces['black'][0]['col']
            if white_square % 2 == black_square % 2:
                return True
        
        return False

    def check_game_end(self) -> Dict[str, Any]:
        """Check if the game has ended."""
        has_moves = self.has_any_valid_moves(self.current_player)
        in_check = self.is_in_check(self.current_player)
        
        result = {'game_over': False, 'winner': None, 'draw_reason': None}
        
        if not has_moves:
            result['game_over'] = True
            if in_check:
                result['winner'] = 'black' if self.current_player == 'white' else 'white'
            else:
                result['draw_reason'] = 'stalemate'
        elif self.is_insufficient_material():
            result['game_over'] = True
            result['draw_reason'] = 'insufficient_material'
        
        return result

    def get_game_state(self) -> GameState:
        """Get the current game state."""
        from models import LastMove
        
        game_end = self.check_game_end()
        
        # Convert last_move format if exists
        last_move_obj = None
        if self.last_move:
            last_move_obj = LastMove(
                from_pos=self.last_move['from'],
                to_pos=self.last_move['to']
            )
        
        return GameState(
            board=self.board,
            current_player=self.current_player,
            move_history=self.move_history,
            captured_by_white=self.captured_by_white,
            captured_by_black=self.captured_by_black,
            last_move=last_move_obj,
            king_positions=self.king_positions,
            castling_rights=self.castling_rights,
            en_passant_target=self.en_passant_target,
            in_check=self.is_in_check(self.current_player),
            game_over=game_end['game_over'],
            winner=game_end['winner'],
            draw_reason=game_end['draw_reason']
        )

    def reset(self):
        """Reset the game."""
        self.init_board()
