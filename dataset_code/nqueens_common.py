"""Core N-Queens logic shared by the dataset generator and the evaluator.

Pure-Python / numpy, no torch dependency so it can be unit-tested and reused
freely. The board is an N x N grid flattened row-major into N*N tokens using the
same value-offset convention as the rest of the puzzle pipeline:

    pad   -> 0   (reserved for the tokenizer/model, unused inside a board)
    empty -> 1   (no queen on this cell)
    queen -> 2   (a queen sits on this cell)

A *puzzle* is a board whose only queens are the given clues (everything else is
``empty``); a *solution* is a full board with exactly N non-attacking queens.
The infill machinery in ``dataloader._infill_view`` treats the queen cells of a
puzzle as the conditioning (held clean) and every empty cell as a blank to fill,
so the model decides queen/empty for all non-clue cells.

8x8 has 92 distinct solutions and 10x10 has 724, so brute-force enumeration of
every solution (and of every completion of a clue set) is cheap.
"""

from functools import lru_cache

import numpy as np

PAD_ID = 0
EMPTY_ID = 1
QUEEN_ID = 2
VOCAB_SIZE = 3  # pad / empty / queen


@lru_cache(maxsize=None)
def enumerate_solutions(n):
    """Return every N-Queens solution as a tuple of (col-per-row) tuples.

    Standard backtracking placing one queen per row. Memoized per ``n`` so the
    (cheap) enumeration runs once. 8x8 -> 92 solutions, 10x10 -> 724.
    """
    solutions = []
    cols = [0] * n

    def place(row, used_cols, used_diag, used_anti):
        if row == n:
            solutions.append(tuple(cols))
            return
        for col in range(n):
            if col in used_cols:
                continue
            d, a = row - col, row + col
            if d in used_diag or a in used_anti:
                continue
            cols[row] = col
            place(row + 1, used_cols | {col}, used_diag | {d}, used_anti | {a})

    place(0, set(), set(), set())
    return tuple(solutions)


def cols_to_board(cols, n):
    """Convert a (col-per-row) assignment into a flat N*N token board."""
    board = np.full(n * n, EMPTY_ID, dtype=np.int64)
    for row, col in enumerate(cols):
        board[row * n + col] = QUEEN_ID
    return board


def board_to_cols(board, n):
    """Inverse of ``cols_to_board``; returns a (col-per-row) tuple.

    Returns ``None`` when any row does not hold exactly one queen, which makes a
    board that violates the one-queen-per-row structure trivially non-matching
    against the enumerated solutions.
    """
    board = np.asarray(board).reshape(n, n)
    cols = []
    for row in range(n):
        queen_cols = np.flatnonzero(board[row] == QUEEN_ID)
        if len(queen_cols) != 1:
            return None
        cols.append(int(queen_cols[0]))
    return tuple(cols)


def is_valid_full_board(board, n):
    """True iff ``board`` holds exactly N mutually non-attacking queens."""
    cols = board_to_cols(board, n)
    if cols is None:
        return False
    if len(set(cols)) != n:  # repeated column
        return False
    diags = {row - col for row, col in enumerate(cols)}
    antis = {row + col for row, col in enumerate(cols)}
    return len(diags) == n and len(antis) == n


def clue_cols_from_board(clue_board, n):
    """Map a clue board to a {row: col} dict of its given queens."""
    board = np.asarray(clue_board).reshape(n, n)
    clues = {}
    for row in range(n):
        queen_cols = np.flatnonzero(board[row] == QUEEN_ID)
        for col in queen_cols:
            clues[row] = int(col)
    return clues


def clue_board_from_cols(clue_cols, n):
    """Build a flat clue board (token grid) from a ``{row: col}`` mapping.

    Inverse of ``clue_cols_from_board``: the given queens become ``queen`` and
    every other cell is ``empty``. Used to materialise an explicit clue set
    (e.g. one chosen by the balanced split) into the puzzle half of an example.
    """
    board = np.full(n * n, EMPTY_ID, dtype=np.int64)
    for row, col in clue_cols.items():
        board[row * n + col] = QUEEN_ID
    return board


def make_puzzle_board(solution_cols, num_clues, rng, n):
    """Reveal ``num_clues`` queens of a full solution as a clue board.

    The kept queens become the puzzle's clues (token ``queen``); every other
    cell is ``empty``. Returns ``(clue_board, clue_cols)`` where ``clue_cols`` is
    the {row: col} mapping of the revealed queens.
    """
    rows = list(range(n))
    rng.shuffle(rows)
    kept_rows = sorted(rows[:num_clues])
    clue_cols = {row: solution_cols[row] for row in kept_rows}
    board = np.full(n * n, EMPTY_ID, dtype=np.int64)
    for row, col in clue_cols.items():
        board[row * n + col] = QUEEN_ID
    return board, clue_cols


def enumerate_completions(clue_cols, n):
    """Every full solution consistent with the given clue queens.

    ``clue_cols`` is a {row: col} mapping. Filters the enumerated solution set,
    which is O(#solutions) and therefore trivial for n in {8, 10}.
    """
    out = []
    for sol in enumerate_solutions(n):
        if all(sol[row] == col for row, col in clue_cols.items()):
            out.append(sol)
    return out


def count_completions(clue_cols, n):
    """Number of full solutions consistent with the clue queens."""
    return len(enumerate_completions(clue_cols, n))


def board_is_consistent_with_clues(board, clue_board, n):
    """True iff every clue queen is also a queen in ``board``."""
    board = np.asarray(board).reshape(-1)
    clue_board = np.asarray(clue_board).reshape(-1)
    clue_cells = np.flatnonzero(clue_board == QUEEN_ID)
    return bool(np.all(board[clue_cells] == QUEEN_ID))
