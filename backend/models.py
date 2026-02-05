"""Pydantic models for the chess API."""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Position(BaseModel):
    row: int
    col: int


class Move(BaseModel):
    from_pos: Position
    to_pos: Position
    en_passant: bool = False
    castling: Optional[str] = None  # 'kingSide' or 'queenSide'


class Piece(BaseModel):
    color: str  # 'white' or 'black'
    type: str   # 'pawn', 'rook', 'knight', 'bishop', 'queen', 'king'


class ValidMove(Position):
    en_passant: bool = False
    castling: Optional[str] = None


class MoveRequest(BaseModel):
    from_row: int
    from_col: int
    to_row: int
    to_col: int
    promotion_piece: Optional[str] = None


class MoveResponse(BaseModel):
    success: bool
    message: str
    game_state: Optional[Dict[str, Any]] = None


class ValidMovesResponse(BaseModel):
    moves: List[ValidMove]


class GameState(BaseModel):
    board: List[List[Optional[Piece]]]
    current_player: str
    move_history: List[Dict[str, Any]]
    captured_by_white: List[Piece]
    captured_by_black: List[Piece]
    last_move: Optional[Move] = None
    king_positions: Dict[str, Position]
    castling_rights: Dict[str, Dict[str, bool]]
    en_passant_target: Optional[Position] = None
    in_check: bool = False
    game_over: bool = False
    winner: Optional[str] = None
    draw_reason: Optional[str] = None


class AIMoveResponse(BaseModel):
    move: Optional[Move] = None
    from_pos: Optional[Position] = None
    to_pos: Optional[Position] = None
    game_state: Optional[GameState] = None
    message: str
