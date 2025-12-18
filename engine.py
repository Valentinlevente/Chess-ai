import time
import chess
import numpy as np
from tensorflow.keras.models import load_model
from pgn_to_matrix import board_to_tensor

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

MODEL_PATH = r"models\policy_model.h5"
model = load_model(MODEL_PATH, compile=False)

# -------------------------------------------------
# MOVE ENCODING 
# -------------------------------------------------

def move_to_index(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    promo = move.promotion

    index = from_sq * 64 + to_sq

    if promo:
        promo_map = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3
        }
        index = 4096 + to_sq * 4 + promo_map[promo]

    return index


def index_to_move(board: chess.Board, index: int) -> chess.Move:
    if index < 4096:
        from_sq = index // 64
        to_sq = index % 64
        return chess.Move(from_sq, to_sq)

    prom_idx = index - 4096
    to_sq = prom_idx // 4
    promo_map = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    promo_piece = promo_map[prom_idx % 4]

    for move in board.legal_moves:
        if move.to_square == to_sq and move.promotion == promo_piece:
            return move

    return None


# -------------------------------------------------
# POLICY MOVE SELECTION 
# -------------------------------------------------

def choose_best_policy_move(board: chess.Board):
    
    tensor = board_to_tensor(board)
    tensor = np.transpose(tensor, (1, 2, 0)) 
    tensor = tensor.reshape(1, 8, 8, 21)

    policy = model.predict(tensor, verbose=0)[0] 

    best_move = None
    best_score = -1

    for move in board.legal_moves:
        idx = move_to_index(move)
        score = policy[idx]

        if score > best_score:
            best_score = score
            best_move = move

    return best_move, best_score


# -------------------------------------------------
# PLAY AGAINST POLICY ENGINE
# -------------------------------------------------

def play_game():
    board = chess.Board()

    print("Starting game!")

    while not board.is_game_over():
        print("\nYour turn (UCI move):")
        print(board)

        # ----- USER MOVE -----
        while True:
            user_input = input("> ").strip()
            try:
                move = chess.Move.from_uci(user_input)
                if move in board.legal_moves:
                    board.push(move)
                    break
                print("Illegal move.")
            except:
                print("Invalid input.")

        if board.is_game_over():
            break

        # ----- ENGINE MOVE -----
        print("\nEngine thinking...")
        move, score = choose_best_policy_move(board)
        print(f"Engine plays: {move}, policy_score={score:.5f}")
        board.push(move)

    print("\nGame over! Result:", board.result())


# -------------------------------------------------
# MAIN
# -------------------------------------------------

if __name__ == "__main__":
    play_game()