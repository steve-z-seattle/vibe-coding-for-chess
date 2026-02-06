"""PGN (Portable Game Notation) parser."""
import copy
import re
from typing import List, Dict, Any, Optional, Tuple
from chess_game import ChessGame
from models import Position, Piece


class PGNGame:
    """Represents a parsed PGN game."""
    
    def __init__(self):
        self.headers: Dict[str, str] = {}
        self.moves: List[str] = []
        self.result: Optional[str] = None


def parse_pgn(pgn_text: str) -> PGNGame:
    """Parse PGN text and return a PGNGame object."""
    game = PGNGame()
    lines = pgn_text.strip().split('\n')
    
    # Parse headers
    move_text_start = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            # Parse header like [Event "FIDE World Championship"]
            match = re.match(r'\[(\w+)\s+"([^"]*)"\]', line)
            if match:
                key, value = match.groups()
                game.headers[key] = value
        elif line and not line.startswith('['):
            move_text_start = i
            break
    
    # Parse move text
    move_lines = ' '.join(lines[move_text_start:])
    game.moves = parse_move_text(move_lines)
    
    # Extract result
    for result in ['1-0', '0-1', '1/2-1/2', '*']:
        if result in move_lines:
            game.result = result
            break
    
    return game


def parse_move_text(move_text: str) -> List[str]:
    """Parse move text into individual moves."""
    # Remove comments { } and ( )
    move_text = re.sub(r'\{[^}]*\}', ' ', move_text)
    move_text = re.sub(r'\([^)]*\)', ' ', move_text)
    
    # Remove move numbers and results
    move_text = re.sub(r'\d+\.', ' ', move_text)  # Remove "1." "2." etc
    move_text = re.sub(r'1-0|0-1|1/2-1/2|\*', ' ', move_text)
    
    # Split into tokens and filter empty ones
    tokens = [t.strip() for t in move_text.split() if t.strip()]
    
    return tokens


def algebraic_to_move(game: ChessGame, alg_move: str) -> Optional[Tuple[Position, Position, Optional[str]]]:
    """
    Convert algebraic notation to internal move format.
    Returns (from_pos, to_pos, promotion_piece) or None if invalid.
    """
    alg_move = alg_move.strip()
    if not alg_move or alg_move in ['1-0', '0-1', '1/2-1/2', '*']:
        return None
    
    # Remove check/checkmate markers
    alg_move = alg_move.replace('+', '').replace('#', '')
    
    # Handle castling
    if alg_move == 'O-O' or alg_move == 'O-O-O':
        return parse_castling(game, alg_move)
    
    # Handle promotion (e.g., e8=Q, exd8=Q)
    promotion = None
    if '=' in alg_move:
        match = re.match(r'(.+)=([QRNB])', alg_move)
        if match:
            alg_move = match.group(1)
            promotion_map = {'Q': 'queen', 'R': 'rook', 'N': 'knight', 'B': 'bishop'}
            promotion = promotion_map.get(match.group(2))
    
    # Determine piece type
    piece_type = 'pawn'
    if alg_move[0] in 'NBRQK':
        piece_map = {'N': 'knight', 'B': 'bishop', 'R': 'rook', 'Q': 'queen', 'K': 'king'}
        piece_type = piece_map[alg_move[0]]
        alg_move = alg_move[1:]
    
    # Check for capture
    is_capture = 'x' in alg_move
    if is_capture:
        alg_move = alg_move.replace('x', '')
    
    # Destination square is always the last 2 characters
    if len(alg_move) < 2:
        return None
    
    dest_file = alg_move[-2]
    dest_rank = alg_move[-1]
    dest_col = ord(dest_file) - ord('a')
    dest_row = 8 - int(dest_rank)
    
    # Disambiguation info (file, rank, or both)
    disambiguation = alg_move[:-2] if len(alg_move) > 2 else ''
    
    # Find the source piece
    from_pos = find_source_square(game, piece_type, dest_row, dest_col, is_capture, disambiguation)
    
    if from_pos is None:
        return None
    
    return (from_pos, Position(row=dest_row, col=dest_col), promotion)


def parse_castling(game: ChessGame, alg_move: str) -> Optional[Tuple[Position, Position, None]]:
    """Parse castling move."""
    color = game.current_player
    row = 7 if color == 'white' else 0
    
    if alg_move == 'O-O':  # King side
        return (Position(row=row, col=4), Position(row=row, col=6), None)
    elif alg_move == 'O-O-O':  # Queen side
        return (Position(row=row, col=4), Position(row=row, col=2), None)
    
    return None


def find_source_square(game: ChessGame, piece_type: str, dest_row: int, dest_col: int, 
                       is_capture: bool, disambiguation: str) -> Optional[Position]:
    """Find the source square for a move given the destination and piece type."""
    color = game.current_player
    
    # Get all pieces of the correct type and color
    candidates = []
    for row in range(8):
        for col in range(8):
            piece = game.board[row][col]
            if piece and piece.color == color and piece.type == piece_type:
                # Check if this piece can move to the destination
                valid_moves = game.get_valid_moves(row, col)
                for move in valid_moves:
                    if move.row == dest_row and move.col == dest_col:
                        candidates.append(Position(row=row, col=col))
                        break
    
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Need disambiguation
    for pos in candidates:
        match = True
        if disambiguation:
            # Check file (column letter)
            if disambiguation[0].isalpha():
                file_col = ord(disambiguation[0]) - ord('a')
                if pos.col != file_col:
                    match = False
            # Check rank (row number)
            if len(disambiguation) > 1 and disambiguation[1].isdigit():
                rank_row = 8 - int(disambiguation[1])
                if pos.row != rank_row:
                    match = False
        if match:
            return pos
    
    return candidates[0] if candidates else None


def import_pgn_to_game(pgn_text: str) -> Optional[ChessGame]:
    """Import a PGN game and return a ChessGame with all moves applied."""
    pgn = parse_pgn(pgn_text)
    
    if not pgn.moves:
        return None
    
    game = ChessGame()
    
    for alg_move in pgn.moves:
        move_data = algebraic_to_move(game, alg_move)
        if move_data is None:
            continue
        
        from_pos, to_pos, promotion = move_data
        
        # Make the move
        piece = game.board[from_pos.row][from_pos.col]
        if not piece:
            continue
        
        captured = game.board[to_pos.row][to_pos.col]
        
        # Execute move
        game.board[to_pos.row][to_pos.col] = piece
        game.board[from_pos.row][from_pos.col] = None
        
        # Update king position
        if piece.type == 'king':
            game.king_positions[piece.color] = Position(row=to_pos.row, col=to_pos.col)
        
        # Handle promotion
        if promotion and piece.type == 'pawn':
            piece.type = promotion
        
        # Handle castling rook move
        if piece.type == 'king' and abs(to_pos.col - from_pos.col) == 2:
            # King side castling
            if to_pos.col == 6:
                rook = game.board[to_pos.row][7]
                game.board[to_pos.row][5] = rook
                game.board[to_pos.row][7] = None
            # Queen side castling
            elif to_pos.col == 2:
                rook = game.board[to_pos.row][0]
                game.board[to_pos.row][3] = rook
                game.board[to_pos.row][0] = None
        
        # Handle en passant capture
        if piece.type == 'pawn' and captured is None and from_pos.col != to_pos.col:
            # En passant - capture the pawn behind
            captured_row = from_pos.row
            captured = game.board[captured_row][to_pos.col]
            game.board[captured_row][to_pos.col] = None
        
        # Track captured pieces
        if captured:
            if piece.color == 'white':
                game.captured_by_white.append(captured)
            else:
                game.captured_by_black.append(captured)
        
        # Update last move
        game.last_move = {
            'from': {'row': from_pos.row, 'col': from_pos.col},
            'to': {'row': to_pos.row, 'col': to_pos.col}
        }
        
        # Switch player
        game.current_player = 'black' if game.current_player == 'white' else 'white'
        
        # Check if the opponent's king is in check (the player whose turn it is now)
        is_check = game.is_in_check(game.current_player)
        
        # Add to move history (store full state for history review)
        game.move_history.append({
            'piece': piece,
            'from': from_pos,
            'to': to_pos,
            'captured': captured,
            'board': copy.deepcopy(game.board),
            'castling_rights': copy.deepcopy(game.castling_rights),
            'king_positions': copy.deepcopy(game.king_positions),
            'en_passant_target': game.en_passant_target,
            'current_player': game.current_player,  # player whose turn is next
            'is_check': is_check,
            'is_checkmate': False
        })
    
    return game
