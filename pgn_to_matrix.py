import os
import numpy as np
import chess
import chess.pgn
from tqdm import tqdm

# ---------------- CONFIG ----------------

PGN_FILE = r"data\raw_pgn\LumbrasGigaBase_OTB_2020-2024.pgn"
OUTPUT_DIR = r"data\training_data"

MAX_POSITIONS = 5_000_000
CHUNK_SIZE   = 250_000

# ---------------- MOVE ENCODING ----------------

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


# ---------------- BOARD TO TENSOR ----------------

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    21 csatornás reprezentáció:
    0-5: white P, N, B, R, Q, K
    6-11: black P, N, B, R, Q, K
    12: side to move (+1 / -1)
    13-16: castling jogok
    17: halfmove clock / 100
    18: white material / 39
    19: black material / 39
    20: legal moves count / 100
    """
    t = np.zeros((21, 8, 8), dtype=np.float32)
    wmat = bmat = 0

    for square, piece in board.piece_map().items():
        r, c = divmod(square, 8)
        idx = piece.piece_type - 1
        if piece.color == chess.BLACK:
            idx += 6
        t[idx, r, c] = 1.0

        val = PIECE_VALUES[piece.piece_type]
        if piece.color:
            wmat += val
        else:
            bmat += val

    t[12] = 1.0 if board.turn else -1.0
    t[13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    t[14] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    t[15] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    t[16] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    t[17] = board.halfmove_clock / 100.0
    t[18] = wmat / 39.0
    t[19] = bmat / 39.0
    t[20] = board.legal_moves.count() / 100.0

    return t


# ---------------- SAVE CHUNK ----------------

def save_chunk(X, Y, chunk_idx):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"policy_dataset_part{chunk_idx}.npz")
    print(f"\nSave chunk{chunk_idx}({len(X)}) to {out_path}")
    X_arr = np.array(X, dtype=np.float32)
    Y_arr = np.array(Y, dtype=np.int32)
    np.savez_compressed(out_path, X=X_arr, Y=Y_arr)
    print("Done")


# ---------------- MAIN ----------------

def main():
    X = []
    Y = []
    total = 0
    chunk_idx = 0

    with open(PGN_FILE, "r", errors="ignore") as pgn:
        pbar = tqdm(total=MAX_POSITIONS, desc="Generating policy samples")

        while total < MAX_POSITIONS:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            board = game.board()

            for move in game.mainline_moves():
                if total >= MAX_POSITIONS:
                    break

                X.append(board_to_tensor(board))
                Y.append(move_to_index(move))

                board.push(move)
                total += 1
                pbar.update(1)

                if len(X) >= CHUNK_SIZE:
                    save_chunk(X, Y, chunk_idx)
                    chunk_idx += 1
                    X.clear()
                    Y.clear()

        pbar.close()

    if X:
        save_chunk(X, Y, chunk_idx)

    print(f"\nTotal positions: {total}")
    print(f"Chunks saved: {chunk_idx + (1 if X else 0)}")


if __name__ == "__main__":
    main()
