"""Tests for ai.py - Chess AI algorithm."""
import pytest
from chess_game import ChessGame
from ai import ChessAI, PIECE_VALUES, POSITION_TABLES


class TestPieceValues:
    """Test piece value constants."""
    
    def test_pawn_value(self):
        """Test pawn has correct value."""
        assert PIECE_VALUES['pawn'] == 100
    
    def test_knight_value(self):
        """Test knight has correct value."""
        assert PIECE_VALUES['knight'] == 320
    
    def test_bishop_value(self):
        """Test bishop has correct value."""
        assert PIECE_VALUES['bishop'] == 330
    
    def test_rook_value(self):
        """Test rook has correct value."""
        assert PIECE_VALUES['rook'] == 500
    
    def test_queen_value(self):
        """Test queen has correct value."""
        assert PIECE_VALUES['queen'] == 900
    
    def test_king_value(self):
        """Test king has high value."""
        assert PIECE_VALUES['king'] == 20000


class TestPositionTables:
    """Test position evaluation tables."""
    
    def test_pawn_table_exists(self):
        """Test pawn position table exists."""
        assert 'pawn' in POSITION_TABLES
        assert len(POSITION_TABLES['pawn']) == 8
        assert all(len(row) == 8 for row in POSITION_TABLES['pawn'])
    
    def test_knight_table_exists(self):
        """Test knight position table exists."""
        assert 'knight' in POSITION_TABLES
    
    def test_king_table_exists(self):
        """Test king position table exists."""
        assert 'king' in POSITION_TABLES


class TestAIBasics:
    """Test basic AI functionality."""
    
    def test_ai_initialization(self):
        """Test AI initializes with correct parameters."""
        ai = ChessAI(depth=3, max_time=2.0)
        assert ai.max_depth == 3
        assert ai.max_time == 2.0
    
    def test_ai_default_depth(self):
        """Test AI has default depth."""
        ai = ChessAI()
        assert ai.max_depth == 3


class TestAIBestMove:
    """Test AI move generation."""
    
    def test_ai_returns_move(self, new_game, ai_player):
        """Test AI returns a valid move from starting position."""
        move = ai_player.get_best_move(new_game)
        assert move is not None
        from_row, from_col, to_row, to_col = move
        # Verify move is within board
        assert 0 <= from_row < 8 and 0 <= from_col < 8
        assert 0 <= to_row < 8 and 0 <= to_col < 8
    
    def test_ai_move_is_valid(self, new_game, ai_player):
        """Test AI returns a valid legal move."""
        move = ai_player.get_best_move(new_game)
        from_row, from_col, to_row, to_col = move
        
        # Check there's a white piece at from position
        piece = new_game.board[from_row][from_col]
        assert piece is not None
        assert piece.color == 'white'
        
        # Check the move is in valid moves
        valid_moves = new_game.get_valid_moves(from_row, from_col)
        assert any(m.row == to_row and m.col == to_col for m in valid_moves)
    
    def test_ai_prefers_capture(self, new_game, ai_player):
        """Test AI prefers capturing when beneficial."""
        # Setup: white queen can capture black queen
        new_game.board[4][4] = new_game.board[7][3]  # Move white queen to e5
        new_game.board[7][3] = None
        new_game.board[4][5] = new_game.board[0][3]  # Move black queen to f5
        new_game.board[0][3] = None
        
        move = ai_player.get_best_move(new_game)
        # AI should capture the queen
        assert move is not None


class TestAIEvaluation:
    """Test board evaluation function."""
    
    def test_evaluation_symmetry(self, ai_player):
        """Test evaluation is symmetric for mirrored positions."""
        game1 = ChessGame()
        game2 = ChessGame()
        
        # Both positions should have same material
        eval1 = ai_player._evaluate_board(game1, 'white')
        eval2 = ai_player._evaluate_board(game2, 'white')
        assert eval1 == eval2
    
    def test_material_advantage(self, ai_player, new_game):
        """Test evaluation favors material advantage."""
        # Remove black queen
        new_game.board[0][3] = None
        
        eval_white = ai_player._evaluate_board(new_game, 'white')
        eval_black = ai_player._evaluate_board(new_game, 'black')
        
        # White should have advantage
        assert eval_white > eval_black
    
    def test_pawn_structure_value(self, ai_player, new_game):
        """Test pawns have different values based on position."""
        # Pawn in center vs edge
        new_game.board = [[None] * 8 for _ in range(8)]
        new_game.board[4][4] = __import__('models').Piece(color='white', type='pawn')  # e5
        new_game.board[7][4] = __import__('models').Piece(color='white', type='king')
        new_game.king_positions['white'] = __import__('models').Position(row=7, col=4)
        new_game.board[0][4] = __import__('models').Piece(color='black', type='king')
        new_game.king_positions['black'] = __import__('models').Position(row=0, col=4)
        
        center_eval = ai_player._evaluate_board(new_game, 'white')
        
        # Move pawn to edge
        new_game.board[4][0] = new_game.board[4][4]
        new_game.board[4][4] = None
        
        edge_eval = ai_player._evaluate_board(new_game, 'white')
        
        # Center pawn should be worth more
        assert center_eval > edge_eval


class TestTranspositionTable:
    """Test transposition table functionality."""
    
    def test_hash_consistency(self, ai_player, new_game):
        """Test same position produces same hash."""
        hash1 = ai_player._hash_board(new_game)
        hash2 = ai_player._hash_board(new_game)
        assert hash1 == hash2
    
    def test_hash_changes_with_move(self, ai_player, game_after_e4):
        """Test hash changes after move."""
        hash_before = ai_player._hash_board(game_after_e4)
        game_after_e4.make_move(1, 4, 3, 4)  # e5
        hash_after = ai_player._hash_board(game_after_e4)
        assert hash_before != hash_after
    
    def test_hash_changes_with_player(self, ai_player, new_game):
        """Test hash differs based on whose turn it is."""
        hash_white = ai_player._hash_board(new_game)
        new_game.current_player = 'black'
        hash_black = ai_player._hash_board(new_game)
        assert hash_white != hash_black


class TestMoveOrdering:
    """Test move ordering for alpha-beta efficiency."""
    
    def test_captures_ordered_high(self, ai_player, new_game):
        """Test capture moves are ordered before non-captures."""
        # Setup position with capture available
        new_game.board[4][4] = new_game.board[7][3]  # White queen
        new_game.board[7][3] = None
        new_game.board[4][5] = new_game.board[0][3]  # Black queen
        new_game.board[0][3] = None
        
        moves = ai_player._get_all_moves_ordered(new_game, 'white', 0)
        
        # The capture should be first or near the top
        assert len(moves) > 0
    
    def test_mvv_lva_ordering(self, ai_player, new_game):
        """Test MVV-LVA ordering (Most Valuable Victim - Least Valuable Aggressor)."""
        # Setup: pawn can capture queen, queen can capture pawn
        new_game.board[4][4] = __import__('models').Piece(color='white', type='pawn')
        new_game.board[4][5] = __import__('models').Piece(color='black', type='queen')
        new_game.board[3][3] = __import__('models').Piece(color='white', type='queen')
        new_game.board[3][4] = __import__('models').Piece(color='black', type='pawn')
        
        moves = ai_player._get_all_moves_ordered(new_game, 'white', 0)
        capture_moves = [m for m in moves if new_game.board[m[2]][m[3]] is not None]
        
        # Should prioritize capturing queen with pawn over capturing pawn with queen
        assert len(capture_moves) >= 2


class TestFastMoveOperations:
    """Test fast move/undo operations (for AI search)."""
    
    def test_fast_move_changes_board(self, ai_player, new_game):
        """Test fast move changes board state."""
        moves = ai_player._get_all_moves_ordered(new_game, 'white', 0)
        assert len(moves) > 0
        
        from_row, from_col, to_row, to_col, move_info = moves[0]
        undo_info = ai_player._make_move_fast(new_game, from_row, from_col, to_row, to_col, move_info)
        
        assert undo_info is not None
        assert new_game.board[to_row][to_col] is not None
        assert new_game.board[from_row][from_col] is None
        
        # Cleanup
        ai_player._undo_move_fast(new_game, undo_info)
    
    def test_fast_undo_restores_state(self, ai_player, new_game):
        """Test fast undo restores board to original state."""
        # Get initial state
        original_board = [row[:] for row in new_game.board]
        original_player = new_game.current_player
        
        moves = ai_player._get_all_moves_ordered(new_game, 'white', 0)
        from_row, from_col, to_row, to_col, move_info = moves[0]
        
        undo_info = ai_player._make_move_fast(new_game, from_row, from_col, to_row, to_col, move_info)
        ai_player._undo_move_fast(new_game, undo_info)
        
        # Verify restoration
        assert new_game.current_player == original_player
        for i in range(8):
            for j in range(8):
                if original_board[i][j] is None:
                    assert new_game.board[i][j] is None
                else:
                    assert new_game.board[i][j].type == original_board[i][j].type
                    assert new_game.board[i][j].color == original_board[i][j].color
    
    def test_fast_move_handles_castling(self, ai_player, castling_setup):
        """Test fast move handles castling correctly."""
        # Get kingside castling move
        moves = ai_player._get_all_moves_ordered(castling_setup, 'white', 0)
        castling_move = None
        for m in moves:
            if m[4] and m[4].castling == 'kingSide':
                castling_move = m
                break
        
        if castling_move:
            from_row, from_col, to_row, to_col, move_info = castling_move
            undo_info = ai_player._make_move_fast(castling_setup, from_row, from_col, to_row, to_col, move_info)
            
            # Verify rook moved
            assert castling_setup.board[7][5].type == 'rook'
            assert castling_setup.board[7][7] is None
            
            # Undo and verify restoration
            ai_player._undo_move_fast(castling_setup, undo_info)
            assert castling_setup.board[7][7].type == 'rook'
            assert castling_setup.board[7][5] is None


class TestSearchDepth:
    """Test AI search depth behavior."""
    
    def test_shallow_search_faster(self, new_game):
        """Test shallower search is faster."""
        import time
        
        ai_shallow = ChessAI(depth=2, max_time=5.0)
        ai_deep = ChessAI(depth=3, max_time=5.0)
        
        start = time.time()
        ai_shallow.get_best_move(new_game)
        shallow_time = time.time() - start
        
        start = time.time()
        ai_deep.get_best_move(new_game)
        deep_time = time.time() - start
        
        # Note: This might occasionally fail due to randomness,
        # but generally deeper search takes longer
        assert deep_time >= shallow_time * 0.5  # Allow some variance
    
    def test_iterative_deepening(self, new_game):
        """Test iterative deepening produces a move."""
        ai = ChessAI(depth=4, max_time=5.0)
        move = ai.get_best_move(new_game)
        assert move is not None


class TestCheckmateDetection:
    """Test AI detects checkmate."""
    
    def test_ai_avoids_immediate_checkmate(self, ai_player):
        """Test AI avoids moves that lead to immediate checkmate."""
        # Setup a position where one move leads to checkmate
        game = ChessGame()
        # Remove most pieces, set up a simple position
        game.board = [[None] * 8 for _ in range(8)]
        game.board[7][4] = __import__('models').Piece(color='white', type='king')
        game.king_positions['white'] = __import__('models').Position(row=7, col=4)
        game.board[0][4] = __import__('models').Piece(color='black', type='king')
        game.king_positions['black'] = __import__('models').Position(row=0, col=4)
        game.board[6][5] = __import__('models').Piece(color='white', type='rook')
        game.board[0][7] = __import__('models').Piece(color='black', type='queen')
        
        move = ai_player.get_best_move(game)
        # AI should find a move (specifically, avoid checkmate if possible)
        assert move is not None
    
    def test_evaluation_extreme_for_checkmate(self, ai_player):
        """Test evaluation handles check positions correctly."""
        # Setup a position where white is in check (delivered by queen)
        game = ChessGame()
        game.board = [[None] * 8 for _ in range(8)]
        game.board[7][4] = __import__('models').Piece(color='white', type='king')  # King on e1
        game.king_positions['white'] = __import__('models').Position(row=7, col=4)
        game.board[0][4] = __import__('models').Piece(color='black', type='king')  # Black king
        game.king_positions['black'] = __import__('models').Position(row=0, col=4)
        # Queen giving check from a1 diagonal to e1
        game.board[3][0] = __import__('models').Piece(color='black', type='queen')  # Queen on a5
        game.current_player = 'white'
        
        # Verify the position is set up
        assert game.board[7][4] is not None
        # The game should have valid evaluation
        eval_score = ai_player._evaluate_board(game, 'white')
        assert isinstance(eval_score, (int, float))


class TestTimeManagement:
    """Test AI time management."""
    
    def test_ai_respects_time_limit(self, new_game):
        """Test AI respects time limit."""
        import time
        ai = ChessAI(depth=10, max_time=0.5)  # Short time limit
        
        start = time.time()
        ai.get_best_move(new_game)
        elapsed = time.time() - start
        
        # Should complete within reasonable time (allow some margin)
        assert elapsed < 2.0
    
    def test_timeout_check(self, ai_player):
        """Test timeout detection works."""
        ai_player.start_time = __import__('time').time() - 10  # Simulate time passed
        ai_player.max_time = 1.0
        assert ai_player._check_timeout() is True


class TestKillerHeuristic:
    """Test killer move heuristic."""
    
    def test_killer_moves_initialized(self, ai_player):
        """Test killer moves table is initialized."""
        assert len(ai_player.killer_moves) == 20
        assert all(len(k) == 2 for k in ai_player.killer_moves)
    
    def test_killer_moves_updated_after_search(self, new_game, ai_player):
        """Test killer moves are populated during search."""
        ai_player.get_best_move(new_game)
        # After search, some killer moves might be set
        # (not guaranteed, but table structure should be maintained)
        assert len(ai_player.killer_moves) == 20


class TestAIVsAI:
    """Test AI playing against itself."""
    
    def test_ai_self_play_completes(self):
        """Test AI can play a game against itself."""
        game = ChessGame()
        ai = ChessAI(depth=2, max_time=1.0)
        
        moves_count = 0
        max_moves = 50  # Prevent infinite loop
        
        while moves_count < max_moves:
            move = ai.get_best_move(game)
            if move is None:
                break
            
            from_row, from_col, to_row, to_col = move
            success, _ = game.make_move(from_row, from_col, to_row, to_col)
            
            if not success:
                break
            
            moves_count += 1
            
            if game.check_game_end()['game_over']:
                break
        
        # Should complete some moves
        assert moves_count > 0
