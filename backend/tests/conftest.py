"""Pytest configuration and fixtures."""
import pytest
import sys
import os
import warnings

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chess_game import ChessGame
from ai import ChessAI


@pytest.fixture
def new_game():
    """Create a new chess game."""
    return ChessGame()


@pytest.fixture
def ai_player():
    """Create an AI player with shallow depth for fast tests."""
    return ChessAI(depth=2, max_time=1.0)


@pytest.fixture
def game_after_e4(new_game):
    """Game after 1.e4"""
    new_game.make_move(6, 4, 4, 4)  # e2-e4
    return new_game


@pytest.fixture
def castling_setup():
    """Setup a position where castling is possible."""
    game = ChessGame()
    # Clear pieces between king and rooks
    game.board[7][5] = None  # f1
    game.board[7][6] = None  # g1
    game.board[7][3] = None  # d1
    game.board[7][2] = None  # c1
    game.board[7][1] = None  # b1
    return game


@pytest.fixture
def checkmate_setup():
    """Setup a checkmate position (Fool's mate)."""
    game = ChessGame()
    # f3
    game.make_move(6, 5, 5, 5)
    # e5
    game.make_move(1, 4, 3, 4)
    # g4
    game.make_move(6, 6, 4, 6)
    # Qh4#
    game.make_move(0, 3, 4, 7)
    return game
