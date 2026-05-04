"""
lace_constructor.py
=====================

Constructive generator for periodic lace grounds. Models how a lacemaker
actually constructs lace: place starts (bobbins) on the leftmost column,
walk threads forward to the right with allowed steps, and close
periodically.

ALGORITHM
---------

Setup: N x M unit cell on a torus (periodic in BOTH axes). Default B = N
starts in column 0, one per row. Each start emits 2 threads, so 2N threads
total.

Each thread is a walk on the universal cover (Z x Z):
  - starts at (col=0, row=r_start)
  - takes single steps from {(0,+1), (0,-1), (+1,0), (+1,+1), (+1,-1)}
  - closes when at (col=M, row congruent to r_start mod N).
    -- May wrap vertically along the way: a thread can step up past
       row N-1 (entering the periodic copy of row 0 in the cell above)
       or down past row 0 (entering row N-1 of the cell below).
  - cannot create off-pin crossings with previously placed thread segments
    (in the universal cover; that's the geometric criterion)
  - cannot cross itself off-pin

Pins (vertices in the resulting graph): equivalence classes of integer
positions (c, r) under (c mod M, r mod N). Threads visit equivalence
classes; visiting a class twice gives degree 4 there.

Output:
  - graph dictionary in atlas-ish format with src/dst/wrap edges
  - matplotlib PNG visualization

USAGE
-----

    python3 -m auxetic_lace.lace_constructor \\
        --n 4 --m 4 --seed 42 \\
        --output-dir /tmp/lace_constructor_output

Bobbins defaults to n (one start per row).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List, Optional, Tuple

# Lazy-imported when --check-workability is used; importing here so the
# module name is bound at module level for the CLI.
try:
    from .lace_workability import is_lace_workable
    from .manufacturability import is_2in2out
    from .planarity import is_planar as is_planar_check
    _HAS_CHECKERS = True
except ImportError:
    _HAS_CHECKERS = False

# Steps allowed at each thread step: (dcol, drow)
ALLOWED_STEPS = [
    (0, +1),    # up (vertical)
    (0, -1),    # down (vertical)
    (+1, 0),    # right
    (+1, +1),   # right-up diagonal
    (+1, -1),   # right-down diagonal
]


# ---------------------------------------------------------------------------
# Geometry: segment crossing test
# ---------------------------------------------------------------------------

def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_cross_off_pin(s1, s2):
    """Return True iff segments s1=(p1, p2) and s2=(p3, p4) cross at a point
    that is NOT a shared endpoint of either segment.
    """
    p1, p2 = s1
    p3, p4 = s2
    o1 = _orient(p1, p2, p3)
    o2 = _orient(p1, p2, p4)
    o3 = _orient(p3, p4, p1)
    o4 = _orient(p3, p4, p2)
    if ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and \
       ((o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0)):
        return True
    return False


# ---------------------------------------------------------------------------
# Walk construction
# ---------------------------------------------------------------------------

def _walk_one_thread(start: Tuple[int, int],
                      n_cols: int, n_rows: int,
                      placed_segments_in_strip: List[Tuple],
                      pin_degree: dict,
                      rng: random.Random,
                      max_steps: int,
                      vert_wrap_limit: int = 2,
                      max_pin_degree: int = 4) -> Optional[List[Tuple[int, int]]]:
    """Walk a single thread on the universal cover.

    The thread starts at (col, row) = start. It must close at column M
    with row congruent to start[1] mod n_rows. The thread may wander
    vertically up to vert_wrap_limit cells in either direction before
    closing.

    Crossing test: segments are placed in absolute (universal-cover)
    coordinates. We check off-pin intersection of every new segment
    against all `placed_segments_in_strip` — these include this thread's
    own past segments AND all prior threads' segments (in the same
    horizontal strip 0..M).

    Note: prior thread segments need to be wrapped into the strip 0..M
    horizontally already by the caller (since we work column 0..M for
    each thread). Vertically they're at the absolute rows where they
    were placed, which is fine because we're testing in R^2.

    Wait — vertical periodicity means a thread we place starting at row
    r might wrap up to row r + N. The next thread starting at row r'
    works in R² and can have its segments cross those of the previous
    thread in the universal cover even if they're "physically separated"
    by a periodic wrap. To do this right, we need to test crossings
    against ALL periodic copies of prior segments.

    For a tractable first version: test against the prior segments AS
    PLACED (not all periodic copies). If we want strict planarity on the
    torus we'd need more work. For now this gives "approximately planar"
    output, sufficient for visual validation.

    Returns the list of pin positions visited (in absolute coords), or
    None on failure.
    """
    target_col = n_cols
    target_row_mod = start[1]  # any row r with r mod n_rows == target_row_mod

    path = [start]
    own_segments: List[Tuple] = []
    cur = start

    # Vertical excursion bound: don't wander more than vert_wrap_limit
    # cells above or below the starting row.
    row_min = start[1] - vert_wrap_limit * n_rows
    row_max = start[1] + vert_wrap_limit * n_rows

    for step_idx in range(max_steps):
        # Closure check: at target column AND row congruent to start row
        if cur[0] == target_col and (cur[1] - start[1]) % n_rows == 0:
            return path

        candidates = []
        for d in ALLOWED_STEPS:
            nxt = (cur[0] + d[0], cur[1] + d[1])
            # Don't overshoot column
            if nxt[0] > target_col:
                continue
            # Don't go past column 0 (impossible with our steps anyway)
            if nxt[0] < 0:
                continue
            # Vertical excursion bound
            if nxt[1] < row_min or nxt[1] > row_max:
                continue
            # If at column n_cols but row not congruent to start, don't
            # accept this — we can't decrease column
            if nxt[0] == target_col and (nxt[1] - start[1]) % n_rows != 0:
                continue
            # Pin degree cap: stepping into nxt adds 2 to its degree
            # (this thread enters and leaves, except at start/end). Use the
            # mod-reduced equivalence class as the pin identity.
            nxt_pin = (nxt[0] % n_cols, nxt[1] % n_rows)
            cur_deg = pin_degree.get(nxt_pin, 0)
            # If this would close the thread (nxt at target column and same
            # row mod N as start), the thread only adds 1 to nxt_pin's degree
            # (incoming only — the start emission already counted out-degree
            # for the start pin). Otherwise it adds 2.
            is_closure = (nxt[0] == n_cols
                           and (nxt[1] - start[1]) % n_rows == 0)
            deg_increment = 1 if is_closure else 2
            if cur_deg + deg_increment > max_pin_degree:
                continue
            # Off-pin crossing check
            new_seg = (cur, nxt)
            crosses = False
            for seg in placed_segments_in_strip:
                if segments_cross_off_pin(new_seg, seg):
                    crosses = True
                    break
            if not crosses:
                for seg in own_segments:
                    if segments_cross_off_pin(new_seg, seg):
                        crosses = True
                        break
            if crosses:
                continue
            candidates.append(nxt)

        if not candidates:
            return None
        nxt = rng.choice(candidates)
        own_segments.append((cur, nxt))
        path.append(nxt)
        cur = nxt

    # Hit step limit
    if cur[0] == target_col and (cur[1] - start[1]) % n_rows == 0:
        return path
    return None


# ---------------------------------------------------------------------------
# Top-level construction
# ---------------------------------------------------------------------------

def construct_lace(n: int, m: int, bobbins: int = None,
                    seed: int = 0,
                    starts: Optional[List[int]] = None,
                    max_steps_per_thread: Optional[int] = None,
                    max_attempts: int = 100,
                    vert_wrap_limit: int = 2,
                    require_4_regular: bool = True,
                    verbose: bool = False) -> Optional[dict]:
    """Construct a periodic lace ground.

    Bobbins defaults to n (one start per row in column 0).
    """
    if bobbins is None:
        bobbins = n
    if max_steps_per_thread is None:
        max_steps_per_thread = 4 * (n + m)

    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt * 1000)
        if starts is None:
            chosen_starts = sorted(rng.sample(range(n), bobbins))
        else:
            chosen_starts = list(starts)

        # Each start emits 2 threads
        thread_starts = []
        for s in chosen_starts:
            thread_starts.append((0, s))
            thread_starts.append((0, s))

        if verbose:
            print(f"Attempt {attempt}: starts={chosen_starts}, "
                  f"{len(thread_starts)} threads")

        placed_segments: List[Tuple] = []
        thread_paths: List[List[Tuple[int, int]]] = []
        success = True
        # Track pin degree as we go (mod-reduced)
        pin_degree = {}
        # Initial: each thread's start contributes +1 to that pin's degree
        # (the out-direction). The same start pin shared by two threads gets
        # +2 from initialization.
        for ts in thread_starts:
            pin = (ts[0] % m, ts[1] % n)
            pin_degree[pin] = pin_degree.get(pin, 0) + 1
        for ti, ts in enumerate(thread_starts):
            path = _walk_one_thread(
                ts, m, n, placed_segments, pin_degree, rng,
                max_steps_per_thread,
                vert_wrap_limit=vert_wrap_limit,
            )
            if path is None:
                if verbose:
                    print(f"  thread {ti} from {ts} FAILED")
                success = False
                break
            thread_paths.append(path)
            for i in range(len(path) - 1):
                placed_segments.append((path[i], path[i + 1]))
                # Update pin_degree for the pin we step into.
                # Each interior pin contributes +2, the closing pin +1.
                nxt = path[i + 1]
                nxt_pin = (nxt[0] % m, nxt[1] % n)
                is_last = (i + 1 == len(path) - 1)
                pin_degree[nxt_pin] = pin_degree.get(nxt_pin, 0) + (
                    1 if is_last else 2)
            if verbose:
                print(f"  thread {ti} from {ts}: length {len(path) - 1}, "
                      f"endpoint {path[-1]}")

        if not success:
            continue

        # Check: every visited pin has degree exactly 4
        if require_4_regular:
            # pin_degree was tracked during generation
            non_four = {p: d for p, d in pin_degree.items() if d != 4}
            if non_four:
                if verbose:
                    print(f"  REJECTED: {len(non_four)} pins not deg-4: "
                          f"{dict(list(non_four.items())[:5])}")
                continue

        return {
            "n_cols": m,
            "n_rows": n,
            "starts": chosen_starts,
            "threads": [list(p) for p in thread_paths],
            "n_threads": len(thread_paths),
            "n_segments": len(placed_segments),
        }

    return None


# ---------------------------------------------------------------------------
# Convert to atlas-style ground dict
# ---------------------------------------------------------------------------

def to_lace_graph_dict(result: dict, name: str = "constructed",
                        family: str = "constructed") -> dict:
    """Convert a construct_lace result into an atlas-style ground dict.

    Pin equivalence: (col, row) ~ (col + k*M, row + j*N) for any integers
    k, j. The vertex set is visited equivalence classes. Each thread
    segment becomes a directed edge with src/dst (vertex indices) and
    wrap (which periodic copy of dst is meant).
    """
    n_cols = result["n_cols"]
    n_rows = result["n_rows"]

    # Find all visited equivalence classes
    visited = set()
    for path in result["threads"]:
        for (c, r) in path:
            visited.add((c % n_cols, r % n_rows))
    visited_pins = sorted(visited)
    pin_to_idx = {p: i for i, p in enumerate(visited_pins)}

    # Build directed edges
    edges = []
    for path in result["threads"]:
        for i in range(len(path) - 1):
            sc, sr = path[i]
            dc, dr = path[i + 1]
            # Wrap: how many cells to shift dst to land in same cell as src
            # The src equivalence class is (sc mod M, sr mod N)
            # The dst equivalence class is (dc mod M, dr mod N)
            # The directed edge in the atlas is from src class to dst class
            # with wrap = (number of M-cells dst is east of src,
            #              number of N-cells dst is north of src)
            wrap_x = (dc // n_cols) - (sc // n_cols)
            # For source side, sr // n_rows gives the cell-row of src.
            wrap_y = (dr // n_rows) - (sr // n_rows)
            # But since both src and dst should be in their own equivalence
            # class, we need to make sure these are computed for the
            # canonical (mod) coords. The canonical src is (sc mod M, sr mod N)
            # in cell (sc // M, sr // N). The canonical dst is at (dc mod M,
            # dr mod N) in cell (dc // M, dr // N). The edge in the atlas
            # convention places src in cell (0, 0), so dst must be in cell
            # (dc//M - sc//M, dr//N - sr//N).
            edges.append({
                "src": pin_to_idx[(sc % n_cols, sr % n_rows)],
                "dst": pin_to_idx[(dc % n_cols, dr % n_rows)],
                "wrap": [wrap_x, wrap_y],
            })

    return {
        "family": family,
        "name": name,
        "n_cols": n_cols,
        "n_rows": n_rows,
        "n_vertices": len(visited_pins),
        "n_edges": len(edges),
        "vertices": [list(p) for p in visited_pins],
        "edges": edges,
        "lattice": [[1.0, 0.0], [0.0, 1.0]],
        "cell_area": float(n_cols * n_rows),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def render_construction(result: dict, output_path: str,
                          n_repeats: int = 3) -> None:
    """Render the constructed lace pattern to a PNG, tiled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = result["n_cols"]
    n_rows = result["n_rows"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Draw periodic pin grid
    for kx in range(-1, n_repeats + 1):
        for ky in range(-1, n_repeats + 1):
            for r in range(n_rows):
                for c in range(n_cols):
                    x = c + kx * n_cols
                    y = r + ky * n_rows
                    ax.plot(x, y, 'o', color='lightgray', markersize=4,
                             zorder=1)

    # Draw threads tiled in both directions
    cmap = plt.get_cmap('tab10')
    for ti, path in enumerate(result["threads"]):
        color = cmap(ti % 10)
        # Thread already contains absolute coords (not mod-reduced).
        # Tile horizontally and vertically.
        for kx in range(-1, n_repeats + 1):
            for ky in range(-1, n_repeats + 1):
                xs = [p[0] + kx * n_cols for p in path]
                ys = [p[1] + ky * n_rows for p in path]
                ax.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.8,
                         zorder=2)

    # Draw the unit cell
    cell_x = [0, n_cols, n_cols, 0, 0]
    cell_y = [0, 0, n_rows, n_rows, 0]
    ax.plot(cell_x, cell_y, '--', color='crimson', linewidth=1.5, alpha=0.8,
             zorder=3)

    ax.set_aspect('equal')
    ax.set_xlim(-1, n_repeats * n_cols + 1)
    ax.set_ylim(-1, n_repeats * n_rows + 1)
    ax.set_title(f"Constructed lace: {n_rows}x{n_cols} cell, "
                  f"{result['n_threads']} threads, "
                  f"starts={result['starts']}")
    ax.grid(True, which='both', alpha=0.2)

    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                 facecolor='#f5f0e3')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Constructive lace generator")
    ap.add_argument("--n", type=int, required=True,
                    help="Number of rows in unit cell")
    ap.add_argument("--m", type=int, required=True,
                    help="Number of columns in unit cell")
    ap.add_argument("--bobbins", type=int, default=None,
                    help="Number of starts. Defaults to n (one per row).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=100)
    ap.add_argument("--vert-wrap-limit", type=int, default=2,
                    help="Max vertical excursion in cell-units (default 2)")
    ap.add_argument("--output-dir", type=str, default=".")
    ap.add_argument("--name", type=str, default="constructed")
    ap.add_argument("--starts", type=str, default=None,
                    help="Comma-separated starting rows (overrides random)")
    ap.add_argument("--no-require-4-regular", action="store_true",
                    help="Allow grounds where some pins have degree != 4 "
                         "(default: require all visited pins to be degree 4)")
    ap.add_argument("--check-workability", action="store_true",
                    help="Run is_lace_workable / is_planar / is_2in2out checks")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.bobbins is None:
        args.bobbins = args.n
    if args.bobbins > args.n:
        print(f"ERROR: bobbins ({args.bobbins}) must be <= n ({args.n})")
        sys.exit(1)

    starts = None
    if args.starts:
        starts = [int(s) for s in args.starts.split(",")]
        if len(starts) != args.bobbins:
            print(f"ERROR: --starts has {len(starts)} entries, "
                  f"--bobbins is {args.bobbins}")
            sys.exit(1)

    print(f"Constructing {args.n}x{args.m} cell with {args.bobbins} starts "
          f"({2 * args.bobbins} threads), seed={args.seed}, "
          f"vert_wrap_limit={args.vert_wrap_limit}...")

    result = construct_lace(
        n=args.n, m=args.m, bobbins=args.bobbins,
        seed=args.seed, starts=starts,
        max_attempts=args.max_attempts,
        vert_wrap_limit=args.vert_wrap_limit,
        require_4_regular=not args.no_require_4_regular,
        verbose=args.verbose,
    )

    if result is None:
        print(f"FAILED to construct after {args.max_attempts} attempts.")
        sys.exit(2)

    print(f"Success: {result['n_threads']} threads placed, "
          f"{result['n_segments']} total segments.")

    os.makedirs(args.output_dir, exist_ok=True)

    graph = to_lace_graph_dict(result, name=args.name)
    graph_path = os.path.join(args.output_dir, "graph.json")
    with open(graph_path, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"Wrote {graph_path}")

    img_path = os.path.join(args.output_dir, "lace.png")
    render_construction(result, img_path)
    print(f"Wrote {img_path}")

    print()
    print(f"Vertices visited: {graph['n_vertices']} (out of "
          f"{args.n * args.m} cell pins)")
    print(f"Edges total: {graph['n_edges']}")
    deg = [0] * graph["n_vertices"]
    for e in graph["edges"]:
        deg[e["src"]] += 1
        deg[e["dst"]] += 1
    from collections import Counter
    deg_counts = Counter(deg)
    for d in sorted(deg_counts):
        print(f"  degree {d}: {deg_counts[d]} vertices")

    if args.check_workability:
        print()
        if not _HAS_CHECKERS:
            print("Cannot run workability check (manufacturability "
                  "module not importable)")
        else:
            print("Manufacturability checks:")
            ok_2in2out = is_2in2out(graph)
            print(f"  is_2in2out:       {ok_2in2out}")
            ok_planar = is_planar_check(graph)
            print(f"  is_planar:        {ok_planar}")
            ok_lace = is_lace_workable(graph) if ok_2in2out else False
            print(f"  is_lace_workable: {ok_lace}")
            if ok_2in2out and ok_planar and ok_lace:
                print()
                print("This is a valid bobbin-lace ground!")


if __name__ == "__main__":
    main()
