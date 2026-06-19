"""N-Queens dataset generator (in-memory, deterministic).

Follows the data-generation procedure of the paper:

  1. Enumerate every complete N-Queens solution (N=8 -> 92, N=10 -> 724).
  2. Build puzzles by *removing* ``k`` queens from a solution; the remaining
     partial board is the input and the original full board is the target.
  3. Use the paper's removal schedule -- ``k in {5,6,7}`` for 8x8 and
     ``k in {7,8,9}`` for 10x10 -- which in both cases leaves ``{1,2,3}`` clue
     queens. Generation is split **equally across k** (an equal number of
     (input, solution) pairs per removal count).
  4. Split 85:15 train/test **by unique input configuration**, so no clue board
     ever appears on both sides (no input leakage). A small validation set is
     carved out of the train side the same way (by unique input).

Boards are flattened row-major into ``N*N`` tokens with vocab pad=0, empty=1,
queen=2; examples are laid out flat as ``[puzzle(N*N) | solution(N*N)]`` (no BOS
/ separators), and ``dataloader._infill_view`` derives the infill view (the clue
queens are the conditioning, every other cell is a blank to fill).

Two entry points share one deterministic partition (same ``seed`` -> same
split), so the training data and the eval set are guaranteed disjoint by input:
  - ``generate_nqueens_dataset`` -> ``{'train', 'validation'}`` in the flat
    tokenized layout, drawn from the 85% train side.
  - ``build_nqueens_eval_puzzles`` -> annotated eval puzzles drawn from the 15%
    test side, each carrying its full completion set and completion count for
    the accuracy / coverage metrics and the GRAM-style plots.
"""

import random
from itertools import combinations

from dataset_code.nqueens_common import (
    enumerate_completions,
    enumerate_solutions,
    clue_board_from_cols,
    cols_to_board,
)

# Fraction of unique input configurations held out as the test set, per the
# paper's 85:15 train/test split.
TEST_FRACTION = 0.15
# Fraction of the *train-side* inputs reserved for the training-time validation
# set (also carved by unique input, so it never leaks into train).
VALID_FRACTION = 0.1


def _paper_remove_counts(n):
    """Number of queens *removed* to form a puzzle, per the paper's schedule.

    8x8 removes ``k in {5,6,7}`` and 10x10 removes ``k in {7,8,9}``; both leave
    ``{1,2,3}`` clue queens. For other board sizes we keep the same low-clue
    regime by removing the three largest counts that still leave >=1 clue.
    """
    if n == 8:
        return [5, 6, 7]
    if n == 10:
        return [7, 8, 9]
    return [n - 3, n - 2, n - 1]


def _default_clue_range(n):
    """Clue counts kept = ``N - k`` over the paper's removal counts -> {1,2,3}."""
    return sorted(n - k for k in _paper_remove_counts(n))


def _input_key(clue_cols):
    """Canonical, hashable key for a clue board (its set of {row: col} queens)."""
    return tuple(sorted(clue_cols.items()))


def _enumerate_pairs(n, num_clues):
    """Every unique ``(input_key, clue_cols, solution_cols)`` revealing exactly
    ``num_clues`` queens of some complete solution.

    Each solution contributes ``C(n, num_clues)`` clue boards; the same clue
    board recurs paired with every solution that contains it, which is precisely
    what lets the model learn the conditional distribution ``p(solution | clues)``
    (a prerequisite for the coverage metric to be meaningful). ``(key, sol)`` is
    unique by construction, so no dedup is needed.
    """
    records = []
    for sol in enumerate_solutions(n):
        for rows in combinations(range(n), num_clues):
            clue_cols = {row: sol[row] for row in rows}
            records.append((_input_key(clue_cols), clue_cols, sol))
    return records


def _split_inputs(records, seed, tag, fraction):
    """Partition pair-records into two sides by *unique input* (clue board).

    All pairs sharing an input land on the same side, so no input crosses the
    boundary. ``fraction`` of the distinct inputs go to the first (held-out)
    side. Deterministic given ``seed``/``tag``.
    """
    by_key = {}
    for rec in records:
        by_key.setdefault(rec[0], []).append(rec)
    keys = sorted(by_key)
    random.Random(f"{seed}:{tag}").shuffle(keys)
    n_held = int(round(len(keys) * fraction))
    if len(keys) > 1:
        n_held = min(max(n_held, 1), len(keys) - 1)  # keep both sides non-empty
    held_keys = set(keys[:n_held])
    held = [rec for k in keys if k in held_keys for rec in by_key[k]]
    rest = [rec for k in keys if k not in held_keys for rec in by_key[k]]
    return held, rest


def _equalize_across_k(by_k, seed, tag, cap_total=None):
    """Take an equal number of pair-records from each clue count (the equal
    split across k). The per-k count is the smallest available across k, further
    capped so the total stays near ``cap_total`` when given. Deterministic.
    """
    clue_counts = sorted(by_k)
    if not clue_counts:
        return []
    per_k = min(len(by_k[nc]) for nc in clue_counts)
    if cap_total:
        per_k = min(per_k, max(1, cap_total // len(clue_counts)))
    out = []
    for nc in clue_counts:
        recs = list(by_k[nc])
        random.Random(f"{seed}:{nc}:{tag}").shuffle(recs)
        out.extend(recs[:per_k])
    return out


def _partition_by_k(n, clue_range, seed, test_fraction):
    """Per clue count, split the full pair universe into train / test sides by
    unique input. Returns ``(train_by_k, test_by_k)`` mapping clue count -> list
    of ``(input_key, clue_cols, solution_cols)`` pair-records.
    """
    train_by_k, test_by_k = {}, {}
    for num_clues in clue_range:
        records = _enumerate_pairs(n, num_clues)
        test_recs, train_recs = _split_inputs(
            records, seed, f"{n}:{num_clues}:test", test_fraction)
        train_by_k[num_clues] = train_recs
        test_by_k[num_clues] = test_recs
    return train_by_k, test_by_k


def _tokenize_records(records, n):
    """Flatten pair-records into the ``[puzzle | solution]`` training layout.

    ``valid_tokens`` is 0 over the puzzle half and 1 over the solution half; the
    infill view is derived downstream.
    """
    half = n * n
    input_ids, valid_tokens = [], []
    for _key, clue_cols, sol in records:
        puzzle = clue_board_from_cols(clue_cols, n)
        solution = cols_to_board(sol, n)
        input_ids.append(list(map(int, puzzle)) + list(map(int, solution)))
        valid_tokens.append([0] * half + [1] * half)
    return {'input_ids': input_ids, 'valid_tokens': valid_tokens}


def generate_nqueens_dataset(n, num_train, num_valid, seed, clue_range=None,
                             test_fraction=TEST_FRACTION,
                             valid_fraction=VALID_FRACTION):
    """Generate train/validation splits in the flat [puzzle | solution] layout.

    The 15% test inputs are held out (used only by ``build_nqueens_eval_puzzles``)
    so train never sees an eval input. The remaining train-side inputs are split
    once more by unique input into train / validation, and each split is then
    balanced equally across the clue counts (k removal values). ``num_train`` and
    ``num_valid`` act as soft caps on the total examples per split; when the
    unique, balanced space is smaller (e.g. 8x8) the splits use what's available.

    Returns ``{'train': {...}, 'validation': {...}}`` where each split is a dict
    with ``input_ids`` and ``valid_tokens`` lists.
    """
    clue_range = clue_range or _default_clue_range(n)
    train_by_k, _test_by_k = _partition_by_k(n, clue_range, seed, test_fraction)

    # Carve validation out of the train side, per clue count, by unique input.
    tr_by_k, va_by_k = {}, {}
    for num_clues, recs in train_by_k.items():
        va_recs, tr_recs = _split_inputs(
            recs, seed, f"{n}:{num_clues}:valid", valid_fraction)
        tr_by_k[num_clues] = tr_recs
        va_by_k[num_clues] = va_recs

    train_recs = _equalize_across_k(tr_by_k, seed, 'train', cap_total=num_train)
    valid_recs = _equalize_across_k(va_by_k, seed, 'valid', cap_total=num_valid)

    k = max(len(clue_range), 1)
    print(f'[nqueens] n={n}: equal split across k -> {len(train_recs) // k} train + '
          f'{len(valid_recs) // k} valid examples per clue count (clues {clue_range}); '
          f'{len(train_recs)} train / {len(valid_recs)} valid total, '
          f'15% of inputs held out for eval.')

    return {
        'train': _tokenize_records(train_recs, n),
        'validation': _tokenize_records(valid_recs, n),
    }


def build_nqueens_eval_puzzles(n, num_puzzles, seed, clue_range=None,
                               test_fraction=TEST_FRACTION):
    """Build annotated eval puzzles from the held-out 15% test inputs.

    Uses the same deterministic partition as ``generate_nqueens_dataset`` (same
    ``seed`` -> same split), so eval puzzles never share an input with the
    training data. Each distinct test input becomes a puzzle annotated with its
    full completion set (for coverage) and completion count (for binning the
    GRAM-style "# of possible solutions" x-axis). Puzzles are bucketed by
    completion count and drawn round-robin so the x-axis is populated across the
    whole range rather than dominated by whichever count is most frequent.

    Important: have clue_range be the same as what you trained on!
    """
    clue_range = clue_range or _default_clue_range(n)
    _train_by_k, test_by_k = _partition_by_k(n, clue_range, seed, test_fraction)
    rng = random.Random(f"{seed}:nqueens-eval")

    # Collapse the test side to distinct inputs and annotate each.
    candidates = {}
    for num_clues in clue_range:
        for key, clue_cols, _sol in test_by_k[num_clues]:
            if key in candidates:
                continue
            completions = enumerate_completions(clue_cols, n)
            solution_cols = completions[rng.randrange(len(completions))]
            candidates[key] = {
                'puzzle_board': clue_board_from_cols(clue_cols, n),
                'solution_board': cols_to_board(solution_cols, n),
                'clue_cols': clue_cols,
                'completions': tuple(completions),
                'solution_count': len(completions),
            }

    # Bucket by completion count and draw round-robin for an even spread.
    buckets = {}
    for item in candidates.values():
        buckets.setdefault(item['solution_count'], []).append(item)
    for items in buckets.values():
        rng.shuffle(items)
    ordered_counts = sorted(buckets)
    selected = []
    while len(selected) < num_puzzles and any(buckets[c] for c in ordered_counts):
        for c in ordered_counts:
            if buckets[c]:
                selected.append(buckets[c].pop())
                if len(selected) >= num_puzzles:
                    break
    return selected
