"""Test suite for the N-Queens core logic, dataset construction and eval metrics.

Run with `pytest tests/test_nqueens.py` or directly `python tests/test_nqueens.py`.
"""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_code import nqueens_common as nq
from dataset_code.generate_nqueens_dataset import (
    build_nqueens_eval_puzzles,
    generate_nqueens_dataset,
)


# --- solution enumeration ---------------------------------------------------

# Known number of distinct solutions to the (unconstrained) N-Queens problem.
# OEIS A000170.
KNOWN_COUNTS = {1: 1, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724}


def test_solution_counts_match_oeis():
    for n, expected in KNOWN_COUNTS.items():
        assert len(nq.enumerate_solutions(n)) == expected, n


def test_enumerated_solutions_are_valid_and_distinct():
    for n in (4, 5, 6, 8):
        sols = nq.enumerate_solutions(n)
        assert len(set(sols)) == len(sols)            # distinct
        for cols in sols:
            board = nq.cols_to_board(cols, n)
            assert nq.is_valid_full_board(board, n)


def test_no_solutions_for_n2_n3():
    assert len(nq.enumerate_solutions(2)) == 0
    assert len(nq.enumerate_solutions(3)) == 0


# --- board <-> token round-trips --------------------------------------------

def test_board_cols_roundtrip():
    n = 8
    for cols in nq.enumerate_solutions(n)[:20]:
        board = nq.cols_to_board(cols, n)
        assert board.shape == (n * n,)
        assert set(np.unique(board)).issubset({nq.EMPTY_ID, nq.QUEEN_ID})
        assert (board == nq.QUEEN_ID).sum() == n
        assert nq.board_to_cols(board, n) == cols


def test_board_to_cols_rejects_bad_rows():
    n = 8
    board = nq.cols_to_board(nq.enumerate_solutions(n)[0], n)
    two_in_row = board.copy()
    two_in_row[0] = nq.QUEEN_ID
    two_in_row[1] = nq.QUEEN_ID          # row 0 now has 2 queens
    assert nq.board_to_cols(two_in_row, n) is None
    empty_row = board.copy()
    qs = np.flatnonzero(empty_row == nq.QUEEN_ID)
    empty_row[qs[0]] = nq.EMPTY_ID        # some row now has 0 queens
    assert nq.board_to_cols(empty_row, n) is None


def test_clue_board_cols_roundtrip():
    n = 8
    sol = nq.enumerate_solutions(n)[0]
    clue_cols = {2: sol[2], 5: sol[5]}
    board = nq.clue_board_from_cols(clue_cols, n)
    assert (board == nq.QUEEN_ID).sum() == 2
    assert nq.clue_cols_from_board(board, n) == clue_cols


# --- constraint checking ----------------------------------------------------

def test_is_valid_rejects_column_and_diagonal_attacks():
    n = 8
    cols = list(nq.enumerate_solutions(n)[0])
    board = nq.cols_to_board(tuple(cols), n)
    assert nq.is_valid_full_board(board, n)

    same_col = cols[:]
    same_col[1] = same_col[0]            # two queens on the same column
    assert not nq.is_valid_full_board(nq.cols_to_board(tuple(same_col), n), n)


def test_is_valid_matches_reference_constraint_semantics():
    """Cross-check against the reference's is_valid_queens definition:
    every row sums to 1, every column sums to 1, and no diagonal/anti-diagonal
    holds >1 queen.
    """
    def reference_valid(board, n):
        b = board.reshape(n, n) == nq.QUEEN_ID
        if not (b.sum(1) == 1).all():
            return False
        if not (b.sum(0) == 1).all():
            return False
        for off in range(-n + 1, n):
            if np.trace(b, offset=off) > 1:
                return False
            if np.trace(np.fliplr(b), offset=off) > 1:
                return False
        return True

    n = 8
    rng = random.Random(0)
    sols = nq.enumerate_solutions(n)
    for cols in sols:
        board = nq.cols_to_board(cols, n)
        assert nq.is_valid_full_board(board, n) == reference_valid(board, n) == True
    for _ in range(200):
        cols = list(rng.choice(sols))
        r = rng.randrange(n)
        cols[r] = rng.randrange(n)
        board = nq.cols_to_board(tuple(cols), n)
        assert nq.is_valid_full_board(board, n) == reference_valid(board, n)


# --- completions (clue-conditioned counting) --------------------------------

def test_completions_empty_and_full():
    n = 8
    assert nq.count_completions({}, n) == 92          # no clues -> all solutions
    sol = nq.enumerate_solutions(n)[0]
    full = {r: sol[r] for r in range(n)}
    assert nq.count_completions(full, n) == 1         # fully specified -> unique


def test_completions_are_supersets_as_clues_grow():
    n = 8
    sol = nq.enumerate_solutions(n)[3]
    prev = None
    for k in range(0, n + 1):
        clue = {r: sol[r] for r in range(k)}
        c = nq.count_completions(clue, n)
        assert c >= 1
        if prev is not None:
            assert c <= prev
        prev = c


def test_consistency_with_clues():
    n = 8
    rng = random.Random(2)
    sol = nq.enumerate_solutions(n)[0]
    clue_board, clue_cols = nq.make_puzzle_board(sol, 3, rng, n)
    sol_board = nq.cols_to_board(sol, n)
    assert nq.board_is_consistent_with_clues(sol_board, clue_board, n)
    other = [s for s in nq.enumerate_solutions(n)
             if any(s[r] != c for r, c in clue_cols.items())][0]
    assert not nq.board_is_consistent_with_clues(
        nq.cols_to_board(other, n), clue_board, n)


# --- accuracy / coverage metric math ----------------------------------------

def _accuracy_coverage(sample_boards, total_solutions, valid_set, n):
    correct = 0
    found = set()
    for board in sample_boards:
        cols = nq.board_to_cols(board, n)
        if cols is not None and cols in valid_set:
            correct += 1
            found.add(cols)
    accuracy = correct / len(sample_boards)
    coverage = len(found) / max(total_solutions, 1)
    return accuracy, coverage


def test_metrics_perfect_distinct_sampler():
    n = 8
    sols = nq.enumerate_solutions(n)
    valid_set = set(sols)
    samples = [nq.cols_to_board(c, n) for c in sols[:20]]
    acc, cov = _accuracy_coverage(samples, len(sols), valid_set, n)
    assert acc == 1.0
    assert abs(cov - 20 / 92) < 1e-9


def test_metrics_all_invalid_sampler():
    n = 8
    sols = nq.enumerate_solutions(n)
    valid_set = set(sols)
    bad = nq.cols_to_board(sols[0], n).copy()
    bad[np.flatnonzero(bad == nq.QUEEN_ID)[0]] = nq.EMPTY_ID
    samples = [bad.copy() for _ in range(20)]
    acc, cov = _accuracy_coverage(samples, len(sols), valid_set, n)
    assert acc == 0.0 and cov == 0.0


def test_metrics_duplicate_valid_samples_count_once_for_coverage():
    n = 8
    sols = nq.enumerate_solutions(n)
    valid_set = set(sols)
    one = nq.cols_to_board(sols[0], n)
    samples = [one.copy() for _ in range(20)]
    acc, cov = _accuracy_coverage(samples, len(sols), valid_set, n)
    assert acc == 1.0
    assert abs(cov - 1 / 92) < 1e-9


# --- dataset generation -----------------------------------------------------

def _clue_boards(split, n):
    """Set of distinct clue boards (puzzle halves / inputs) in a tokenized split."""
    half = n * n
    return {tuple(int(v) for v in np.asarray(ids)[:half])
            for ids in split["input_ids"]}


def _clue_count(clue_board):
    return sum(int(v) == nq.QUEEN_ID for v in clue_board)


def test_train_examples_are_valid_and_clue_consistent():
    n = 8
    ds = generate_nqueens_dataset(n, num_train=500, num_valid=100, seed=7)
    for ids in ds["train"]["input_ids"]:
        ids = np.asarray(ids)
        half = len(ids) // 2
        puzzle, solution = ids[:half], ids[half:]
        assert len(ids) == 2 * n * n
        assert nq.is_valid_full_board(solution, n)
        assert nq.board_is_consistent_with_clues(solution, puzzle, n)


def test_train_learns_multimodal_clue_distribution():
    """A clue config with many completions must appear paired with >1 distinct
    solution in the training data, otherwise coverage is unlearnable.
    """
    n = 8
    ds = generate_nqueens_dataset(n, num_train=8000, num_valid=500, seed=11)
    sols_per_clue = {}
    half = n * n
    for ids in ds["train"]["input_ids"]:
        ids = np.asarray(ids)
        clue = tuple(int(v) for v in ids[:half])
        sol = nq.board_to_cols(ids[half:], n)
        sols_per_clue.setdefault(clue, set()).add(sol)
    multimodal = [c for c, s in sols_per_clue.items()
                  if _clue_count(c) <= 2 and len(s) > 1]
    assert multimodal, "expected some sparse clue configs with multiple targets"


def test_train_valid_disjoint_by_input():
    """No clue board (input) may appear in both train and validation."""
    n = 8
    ds = generate_nqueens_dataset(n, num_train=2000, num_valid=300, seed=13)
    train = _clue_boards(ds["train"], n)
    valid = _clue_boards(ds["validation"], n)
    assert train and valid
    assert train.isdisjoint(valid)


def test_train_is_balanced_equally_across_k():
    """The train split holds an equal number of examples for each clue count
    (the equal split across k = {5,6,7} removed -> {1,2,3} clues kept).
    """
    n = 8
    ds = generate_nqueens_dataset(n, num_train=40000, num_valid=2000, seed=3)
    per_k = {}
    half = n * n
    for ids in ds["train"]["input_ids"]:
        k = _clue_count(np.asarray(ids)[:half])
        per_k[k] = per_k.get(k, 0) + 1
    assert set(per_k) == {1, 2, 3}, per_k
    assert len(set(per_k.values())) == 1, per_k          # equal count per k
    assert 0 < sum(per_k.values()) <= 24000


def test_train_and_eval_inputs_are_disjoint():
    """Training data and the eval set share no input (same seeded 85:15 split)."""
    n = 8
    ds = generate_nqueens_dataset(n, num_train=2000, num_valid=300, seed=21)
    train = _clue_boards(ds["train"], n) | _clue_boards(ds["validation"], n)
    eval_inputs = {tuple(int(v) for v in p["puzzle_board"])
                   for p in build_nqueens_eval_puzzles(n, 60, seed=21)}
    assert eval_inputs
    assert train.isdisjoint(eval_inputs)


def test_train_only_uses_paper_clue_counts():
    n = 8
    ds = generate_nqueens_dataset(n, num_train=2000, num_valid=300, seed=4)
    half = n * n
    counts = {_clue_count(np.asarray(ids)[:half])
              for ids in ds["train"]["input_ids"]}
    assert counts == {1, 2, 3}, counts            # k in {5,6,7} removed


# --- eval puzzle construction -----------------------------------------------

def test_eval_puzzles_span_solution_counts_and_are_consistent():
    n = 8
    puzzles = build_nqueens_eval_puzzles(n, num_puzzles=60, seed=1)
    assert len(puzzles) == 60
    counts = {p["solution_count"] for p in puzzles}
    assert len(counts) >= 5, "eval set should span several solution counts"
    for p in puzzles:
        assert p["solution_count"] == len(p["completions"])
        for cols in p["completions"]:
            board = nq.cols_to_board(cols, n)
            assert nq.is_valid_full_board(board, n)
            assert nq.board_is_consistent_with_clues(board, p["puzzle_board"], n)


def test_eval_completion_set_is_exactly_the_valid_completions():
    n = 8
    puzzles = build_nqueens_eval_puzzles(n, num_puzzles=30, seed=5)
    all_sols = set(nq.enumerate_solutions(n))
    for p in puzzles:
        clue_cols = nq.clue_cols_from_board(p["puzzle_board"], n)
        brute = {s for s in all_sols
                 if all(s[r] == c for r, c in clue_cols.items())}
        assert set(p["completions"]) == brute


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    sys.exit(1 if failures else 0)
