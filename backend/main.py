"""FastAPI backend for the chess game."""
import os
import asyncio
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from models import (
    MoveRequest, MoveResponse, ValidMovesResponse, ValidMove,
    GameState, AIMoveResponse, Position, AIConfigRequest
)
from chess_game import ChessGame
from ai import ChessAI

# Server version - increment this when making changes
VERSION = "1.2.3"

app = FastAPI(title="Chess API", version=VERSION)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global game store (in production, use a database or session management)
games: Dict[str, ChessGame] = {}
ai_players: Dict[str, ChessAI] = {}
game_locks: Dict[str, asyncio.Lock] = {}

def get_game_lock(game_id: str) -> asyncio.Lock:
    """Get or create a lock for a game."""
    if game_id not in game_locks:
        game_locks[game_id] = asyncio.Lock()
    return game_locks[game_id]


def get_or_create_game(game_id: str = "default") -> ChessGame:
    """Get or create a game instance."""
    if game_id not in games:
        games[game_id] = ChessGame()
        # Default: medium difficulty with 1.5 second time limit
        ai_players[game_id] = ChessAI(depth=3, max_time=1.5)
    return games[game_id]


def game_state_to_dict(game: ChessGame) -> dict:
    """Convert game state to dictionary for JSON response."""
    state = game.get_game_state()
    
    # Convert board to serializable format
    serializable_board = []
    for row in state.board:
        serializable_row = []
        for piece in row:
            if piece:
                serializable_row.append({"color": piece.color, "type": piece.type})
            else:
                serializable_row.append(None)
        serializable_board.append(serializable_row)
    
    # Convert captured pieces
    captured_by_white = [{"color": p.color, "type": p.type} for p in state.captured_by_white]
    captured_by_black = [{"color": p.color, "type": p.type} for p in state.captured_by_black]
    
    # Convert king positions
    king_positions = {
        "white": {"row": state.king_positions["white"].row, "col": state.king_positions["white"].col},
        "black": {"row": state.king_positions["black"].row, "col": state.king_positions["black"].col}
    }
    
    # Convert last move
    last_move = None
    if state.last_move:
        last_move = {
            "from": {"row": state.last_move.from_pos.row, "col": state.last_move.from_pos.col},
            "to": {"row": state.last_move.to_pos.row, "col": state.last_move.to_pos.col}
        }
    
    # Convert en passant target
    en_passant = None
    if state.en_passant_target:
        en_passant = {"row": state.en_passant_target.row, "col": state.en_passant_target.col}
    
    return {
        "board": serializable_board,
        "current_player": state.current_player,
        "move_history": state.move_history,
        "captured_by_white": captured_by_white,
        "captured_by_black": captured_by_black,
        "last_move": last_move,
        "king_positions": king_positions,
        "castling_rights": state.castling_rights,
        "en_passant_target": en_passant,
        "in_check": state.in_check,
        "game_over": state.game_over,
        "winner": state.winner,
        "draw_reason": state.draw_reason
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Chess API is running", "docs": "/docs", "version": VERSION}


@app.get("/api/version")
async def get_version():
    """Get server version."""
    return {"version": VERSION}


@app.get("/api/game/{game_id}/state")
async def get_game_state(game_id: str = "default"):
    """Get the current game state."""
    game = get_or_create_game(game_id)
    return game_state_to_dict(game)


@app.get("/api/game/{game_id}/history/{move_number}")
async def get_game_state_at_move(game_id: str = "default", move_number: int = 0):
    """Get the game state at a specific move number (for review)."""
    game = get_or_create_game(game_id)
    
    # move_number 0 = initial position, 1 = after first move, etc.
    if move_number < 0 or move_number > len(game.move_history):
        raise HTTPException(status_code=400, detail="Invalid move number")
    
    if move_number == 0:
        # Return initial state (standard chess starting position)
        initial_board = [
            [{"color": "black", "type": "rook"}, {"color": "black", "type": "knight"}, {"color": "black", "type": "bishop"}, {"color": "black", "type": "queen"}, {"color": "black", "type": "king"}, {"color": "black", "type": "bishop"}, {"color": "black", "type": "knight"}, {"color": "black", "type": "rook"}],
            [{"color": "black", "type": "pawn"} for _ in range(8)],
            [None] * 8,
            [None] * 8,
            [None] * 8,
            [None] * 8,
            [{"color": "white", "type": "pawn"} for _ in range(8)],
            [{"color": "white", "type": "rook"}, {"color": "white", "type": "knight"}, {"color": "white", "type": "bishop"}, {"color": "white", "type": "queen"}, {"color": "white", "type": "king"}, {"color": "white", "type": "bishop"}, {"color": "white", "type": "knight"}, {"color": "white", "type": "rook"}]
        ]
        return {
            "board": initial_board,
            "current_player": "white",
            "move_history": [],
            "captured_by_white": [],
            "captured_by_black": [],
            "last_move": None,
            "king_positions": {
                "white": {"row": 7, "col": 4},
                "black": {"row": 0, "col": 4}
            },
            "castling_rights": {
                "white": {"kingSide": True, "queenSide": True},
                "black": {"kingSide": True, "queenSide": True}
            },
            "en_passant_target": None,
            "in_check": False,
            "game_over": False,
            "winner": None,
            "draw_reason": None,
            "is_review": True,
            "review_move_number": 0,
            "total_moves": len(game.move_history)
        }
    
    # Get state from move history
    move_data = game.move_history[move_number - 1]
    
    # Calculate captured pieces up to this move
    captured_by_white = []
    captured_by_black = []
    for i in range(move_number):
        move = game.move_history[i]
        if move['captured']:
            if move['piece'].color == 'white':
                captured_by_white.append({"color": move['captured'].color, "type": move['captured'].type})
            else:
                captured_by_black.append({"color": move['captured'].color, "type": move['captured'].type})
    
    # Get last move
    last_move = None
    if move_number >= 1:
        last_move = {
            "from": {"row": move_data['from'].row, "col": move_data['from'].col},
            "to": {"row": move_data['to'].row, "col": move_data['to'].col}
        }
    
    # Convert board state at that move
    board_at_move = move_data['board']
    serializable_board = []
    for row in board_at_move:
        serializable_row = []
        for piece in row:
            if piece:
                serializable_row.append({"color": piece.color, "type": piece.type})
            else:
                serializable_row.append(None)
        serializable_board.append(serializable_row)
    
    return {
        "board": serializable_board,
        "current_player": 'black' if move_data['current_player'] == 'white' else 'white',
        "move_history": game.move_history[:move_number],
        "captured_by_white": captured_by_white,
        "captured_by_black": captured_by_black,
        "last_move": last_move,
        "king_positions": {
            "white": {"row": move_data['king_positions']['white'].row, "col": move_data['king_positions']['white'].col},
            "black": {"row": move_data['king_positions']['black'].row, "col": move_data['king_positions']['black'].col}
        },
        "castling_rights": move_data['castling_rights'],
        "en_passant_target": {"row": move_data['en_passant_target'].row, "col": move_data['en_passant_target'].col} if move_data['en_passant_target'] else None,
        "in_check": False,  # Simplified - would need to calculate
        "game_over": False,
        "winner": None,
        "draw_reason": None,
        "is_review": True,
        "review_move_number": move_number,
        "total_moves": len(game.move_history)
    }


@app.post("/api/game/{game_id}/reset")
async def reset_game(game_id: str = "default"):
    """Reset the game."""
    async with get_game_lock(game_id):
        games[game_id] = ChessGame()
        ai_players[game_id] = ChessAI(depth=3, max_time=1.5)
        return game_state_to_dict(games[game_id])


@app.post("/api/game/{game_id}/move")
async def make_move(move: MoveRequest, game_id: str = "default"):
    """Make a move on the board."""
    async with get_game_lock(game_id):
        game = get_or_create_game(game_id)
        
        success, message = game.make_move(
            move.from_row, move.from_col,
            move.to_row, move.to_col,
            move.promotion_piece
        )
        
        return MoveResponse(
            success=success,
            message=message,
            game_state=game_state_to_dict(game) if success else None
        )


@app.get("/api/game/{game_id}/valid-moves")
async def get_valid_moves(row: int, col: int, game_id: str = "default"):
    """Get valid moves for a piece at the given position."""
    game = get_or_create_game(game_id)
    moves = game.get_valid_moves(row, col)
    
    return ValidMovesResponse(moves=moves)



@app.post("/api/game/{game_id}/ai-move")
async def ai_move(game_id: str = "default"):
    """Make an AI move."""
    async with get_game_lock(game_id):
        game = get_or_create_game(game_id)
        ai = ai_players.get(game_id, ChessAI(depth=3))
        
        # Store current player before AI calculation
        current_player_before = game.current_player
        
        best_move = ai.get_best_move(game)
        
        if not best_move:
            return AIMoveResponse(message="No valid moves available")
        
        from_row, from_col, to_row, to_col = best_move
        
        # Double-check it's still the same player's turn
        if game.current_player != current_player_before:
            return AIMoveResponse(message="Game state changed, please retry")
        
        success, message = game.make_move(from_row, from_col, to_row, to_col)
        
        if not success:
            return AIMoveResponse(message=f"AI move failed: {message}")
        
        return AIMoveResponse(
            move={
                "from": {"row": from_row, "col": from_col},
                "to": {"row": to_row, "col": to_col}
            },
            from_pos=Position(row=from_row, col=from_col),
            to_pos=Position(row=to_row, col=to_col),
            game_state=game_state_to_dict(game),
            message="AI move successful"
        )


@app.get("/api/game/{game_id}/check")
async def check_game_end(game_id: str = "default"):
    """Check if the game has ended."""
    game = get_or_create_game(game_id)
    result = game.check_game_end()
    return result


@app.post("/api/game/{game_id}/ai-config")
async def configure_ai(config: AIConfigRequest, game_id: str = "default"):
    """Configure AI difficulty."""
    global ai_players
    # Map depth to appropriate time limits for better UX
    time_limits = {1: 0.5, 2: 1.0, 3: 1.5, 4: 3.0}  # Easy: 0.5s, Medium: 1s, Hard: 1.5s, Very Hard: 3s
    max_time = time_limits.get(config.depth, 2.0)
    ai_players[game_id] = ChessAI(depth=config.depth, max_time=max_time)
    return {"success": True, "depth": config.depth, "max_time": max_time}


# Mount static files - 从 backend 目录向上退一级到项目根目录，再进入 frontend/static
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
