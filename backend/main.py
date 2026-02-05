"""FastAPI backend for the chess game."""
import os
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from models import (
    MoveRequest, MoveResponse, ValidMovesResponse, ValidMove,
    GameState, AIMoveResponse, Position
)
from chess_game import ChessGame
from ai import ChessAI

app = FastAPI(title="Chess API", version="1.0.0")

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


def get_or_create_game(game_id: str = "default") -> ChessGame:
    """Get or create a game instance."""
    if game_id not in games:
        games[game_id] = ChessGame()
        ai_players[game_id] = ChessAI(depth=3)
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
            "from": {"row": state.last_move["from"].row, "col": state.last_move["from"].col},
            "to": {"row": state.last_move["to"].row, "col": state.last_move["to"].col}
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
    return {"message": "Chess API is running", "docs": "/docs"}


@app.get("/api/game/{game_id}/state")
async def get_game_state(game_id: str = "default"):
    """Get the current game state."""
    game = get_or_create_game(game_id)
    return game_state_to_dict(game)


@app.post("/api/game/{game_id}/reset")
async def reset_game(game_id: str = "default"):
    """Reset the game."""
    games[game_id] = ChessGame()
    ai_players[game_id] = ChessAI(depth=3)
    return game_state_to_dict(games[game_id])


@app.post("/api/game/{game_id}/move")
async def make_move(move: MoveRequest, game_id: str = "default"):
    """Make a move on the board."""
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


@app.post("/api/game/{game_id}/undo")
async def undo_move(game_id: str = "default"):
    """Undo the last move."""
    game = get_or_create_game(game_id)
    success = game.undo_move()
    
    if not success:
        raise HTTPException(status_code=400, detail="Cannot undo move")
    
    return game_state_to_dict(game)


@app.post("/api/game/{game_id}/ai-move")
async def ai_move(game_id: str = "default"):
    """Make an AI move."""
    game = get_or_create_game(game_id)
    ai = ai_players.get(game_id, ChessAI(depth=3))
    
    best_move = ai.get_best_move(game)
    
    if not best_move:
        return AIMoveResponse(message="No valid moves available")
    
    from_row, from_col, to_row, to_col = best_move
    
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


# Mount static files - 从 backend 目录向上退一级到项目根目录，再进入 frontend/static
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
