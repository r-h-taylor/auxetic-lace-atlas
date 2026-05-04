"""
bobbin_lace_constructor.py
============================

Constructive generator that produces grounds satisfying bobbin lace
workability — extending the basic lace_constructor.py by adding a
"kiss-compatible" rotational arrangement check at each pin.

WHY THIS EXISTS
---------------

The basic constructor (lace_constructor.py) produces 4-regular periodic
graphs that are 3D-printable, but the osculating-circuit decomposition
of the resulting graph often glues all our placed threads into one
combined long circuit because of TRANSVERSE arrangements at pin meetings:
two threads going straight through each other rather than each curving
to its own side of the pin.

Real bobbin lace can never have transverse arrangements physically — the
bobbin manipulations always produce KISS arrangements where threads
curve around the pin without crossing each other through it. Mathematically,
this is the rotational pattern [in, in, out, out] (some rotation), AND
additionally each thread's two ends must be rotationally adjacent to
each other (not separated by the other thread's ends).

This generator enforces the kiss-compatible arrangement during walks.
The result is grounds that pass is_lace_workable.

ALGORITHM
---------

Same as lace_constructor.py, plus: when placing a step that would land
at a pin already visited by another thread, check that the new arrangement
of edge-ends at that pin keeps each thread's two ends rotationally
adjacent. Reject any step that would create a transverse arrangement.

Equivalently: at every pin, all 4 edge-ends are tagged with thread_id
and role (in/out). When sorted by angle, the cyclic order should be
[A_*, A_*, B_*, B_*] (some rotation), where A_* are the two ends of
thread A and B_* the two ends of thread B. The transverse pattern
[A_*, B_*, A_*, B_*] is rejected.

USAGE
-----

    python3 -m auxetic_lace.bobbin_lace_constructor \\
        --n 4 --m 4 --seed 42 \\
        --output-dir /tmp/bobbin_out \\
        --check-workability

Same CLI as lace_constructor.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

try:
    from .lace_workability import is_lace_workable
    from .manufacturability import is_2in2out
    from .planarity import is_planar as is_planar_check
    _HAS_CHECKERS = True
except ImportError:
    _HAS_CHECKERS = False


ALLOWED_STEPS = [
    (0, +1),
    (0, -1),
    (+1, 0),
    (+1, +1),
    (+1, -1),
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_cross_off_pin(s1, s2):
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
# Pin-end bookkeeping for kiss compatibility
# ---------------------------------------------------------------------------

# At each pin (canonical (col, row) mod (M, N)), we track a list of
# edge-ends: (angle, role, thread_id). The angle is the angular direction
# from the pin toward the OTHER end of the segment (the direction the
# thread leaves/arrives along).

def _angle(d_col: int, d_row: int) -> float:
    """Angular direction (CCW from +x). Args are direction COMPONENTS."""
    return math.atan2(d_row, d_col)


def _check_kiss_compatible_at_pin(ends: List[Tuple[float, str, int]]) -> bool:
    """Check that the cyclic ordering of edge-ends at a pin is
    kiss-compatible.

    Two checks:

    1. Role pattern: when sorted by angle, the cyclic sequence of roles
       must be [in, in, out, out] (some rotation), not [in, out, in, out]
       (transverse). This is Irvine's vertex consecutivity criterion.

    2. Thread identity: each thread's edge-ends must be rotationally
       adjacent (not interleaved with the other thread's ends).

    For < 4 ends, returns True (partial state during construction).
    """
    if len(ends) < 4:
        return True

    # Sort by angle (CCW)
    sorted_ends = sorted(ends, key=lambda x: x[0])
    roles = [r for (_, r, _) in sorted_ends]
    n = len(roles)

    # Check 1: role pattern must be [in, in, out, out] (some rotation).
    # That is: each role's positions must be rotationally adjacent.
    in_positions = [i for i, r in enumerate(roles) if r == "in"]
    out_positions = [i for i, r in enumerate(roles) if r == "out"]
    for positions in (in_positions, out_positions):
        if len(positions) <= 1:
            continue
        gaps = []
        for i in range(len(positions)):
            cur = positions[i]
            nxt = positions[(i + 1) % len(positions)]
            gap = (nxt - cur) % n
            gaps.append(gap)
        big_gaps = sum(1 for g in gaps if g > 1)
        if big_gaps > 1:
            return False  # role pattern is transverse

    # Check 2: thread-identity adjacency (already correct in original)
    from collections import defaultdict
    positions_by_thread: Dict[int, List[int]] = defaultdict(list)
    for idx, (_, _, tid) in enumerate(sorted_ends):
        positions_by_thread[tid].append(idx)
    for tid, positions in positions_by_thread.items():
        if len(positions) <= 1:
            continue
        positions.sort()
        gaps = []
        for i in range(len(positions)):
            cur = positions[i]
            nxt = positions[(i + 1) % len(positions)]
            gap = (nxt - cur) % n
            gaps.append(gap)
        big_gaps = sum(1 for g in gaps if g > 1)
        if big_gaps > 1:
            return False  # transverse: thread's ends are split across gaps

    return True


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------

def _walk_one_thread(thread_id: int,
                      start: Tuple[int, int],
                      n_cols: int, n_rows: int,
                      placed_segments: List[Tuple],
                      pin_degree: dict,
                      pin_ends: Dict[Tuple[int, int], List[Tuple[float, str, int]]],
                      rng: random.Random,
                      max_steps: int,
                      vert_wrap_limit: int = 2,
                      max_pin_degree: int = 4) -> Optional[List[Tuple[int, int]]]:
    """Walk a single thread on the universal cover with kiss-compatible
    enforcement at every visited pin."""
    target_col = n_cols
    path = [start]
    own_segments: List[Tuple] = []
    cur = start

    # Initial: thread enters its start pin from "outside" — represent
    # this as an out-direction at the start pin. We'll add this end to
    # pin_ends now (the very first step out of start contributes one out).
    # Note: the start pin's first out-end is added when we take the first
    # step (so we know the angle). We pre-create a placeholder for now.

    row_min = start[1] - vert_wrap_limit * n_rows
    row_max = start[1] + vert_wrap_limit * n_rows

    # Track which pin_ends we added during this walk so we can roll back
    # on failure or before close.
    added_ends: List[Tuple[Tuple[int, int], int]] = []

    def add_end(pin_canon, end):
        pin_ends.setdefault(pin_canon, []).append(end)
        added_ends.append((pin_canon, len(pin_ends[pin_canon]) - 1))

    def rollback_ends():
        for pin_canon, idx in reversed(added_ends):
            # Best-effort — we appended in order, so the last entry should
            # correspond to the latest add. But because we may have added
            # multiple to the same pin, just remove from end.
            if pin_ends[pin_canon]:
                pin_ends[pin_canon].pop()

    for step_idx in range(max_steps):
        if cur[0] == target_col and (cur[1] - start[1]) % n_rows == 0:
            return path

        candidates = []
        for d in ALLOWED_STEPS:
            nxt = (cur[0] + d[0], cur[1] + d[1])
            if nxt[0] > target_col:
                continue
            if nxt[0] < 0:
                continue
            if nxt[1] < row_min or nxt[1] > row_max:
                continue
            if nxt[0] == target_col and (nxt[1] - start[1]) % n_rows != 0:
                continue

            # Pin degree cap
            nxt_pin = (nxt[0] % n_cols, nxt[1] % n_rows)
            cur_deg = pin_degree.get(nxt_pin, 0)
            is_closure = (nxt[0] == target_col
                           and (nxt[1] - start[1]) % n_rows == 0)
            deg_increment = 1 if is_closure else 2
            if cur_deg + deg_increment > max_pin_degree:
                continue

            # Off-pin crossing check (with periodic copies)
            new_seg = (cur, nxt)
            crosses = False
            for seg in placed_segments + own_segments:
                for kx in (-1, 0, 1):
                    if crosses:
                        break
                    for ky in (-1, 0, 1):
                        shifted = (
                            (seg[0][0] + kx * n_cols,
                             seg[0][1] + ky * n_rows),
                            (seg[1][0] + kx * n_cols,
                             seg[1][1] + ky * n_rows),
                        )
                        if segments_cross_off_pin(new_seg, shifted):
                            crosses = True
                            break
                if crosses:
                    break
            if crosses:
                continue

            # Kiss-compatible check at both endpoints of the new segment
            cur_canon = (cur[0] % n_cols, cur[1] % n_rows)
            cur_out_angle = _angle(d[0], d[1])
            nxt_in_angle = _angle(-d[0], -d[1])
            # Tentatively add ends and check
            cur_ends_test = list(pin_ends.get(cur_canon, []))
            cur_ends_test.append((cur_out_angle, "out", thread_id))
            if not _check_kiss_compatible_at_pin(cur_ends_test):
                continue
            nxt_ends_test = list(pin_ends.get(nxt_pin, []))
            nxt_ends_test.append((nxt_in_angle, "in", thread_id))
            if not _check_kiss_compatible_at_pin(nxt_ends_test):
                continue

            candidates.append((nxt, cur_out_angle, nxt_in_angle, cur_canon,
                                nxt_pin))

        if not candidates:
            return None

        # Pick one
        chosen = rng.choice(candidates)
        nxt, cur_out_angle, nxt_in_angle, cur_canon, nxt_pin = chosen

        own_segments.append((cur, nxt))
        # Commit the ends to pin_ends
        add_end(cur_canon, (cur_out_angle, "out", thread_id))
        add_end(nxt_pin, (nxt_in_angle, "in", thread_id))
        path.append(nxt)
        cur = nxt

    if cur[0] == target_col and (cur[1] - start[1]) % n_rows == 0:
        return path
    return None


# ---------------------------------------------------------------------------
# Top-level construction
# ---------------------------------------------------------------------------

def construct_bobbin_lace(n: int, m: int, bobbins: Optional[int] = None,
                            seed: int = 0,
                            starts: Optional[List[int]] = None,
                            max_steps_per_thread: Optional[int] = None,
                            max_attempts: int = 100,
                            vert_wrap_limit: int = 2,
                            require_4_regular: bool = True,
                            verbose: bool = False) -> Optional[dict]:
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

        thread_starts = []
        for s in chosen_starts:
            thread_starts.append((0, s))
            thread_starts.append((0, s))

        if verbose:
            print(f"Attempt {attempt}: starts={chosen_starts}, "
                  f"{len(thread_starts)} threads")

        placed_segments: List[Tuple] = []
        thread_paths: List[List[Tuple[int, int]]] = []
        pin_degree: Dict[Tuple[int, int], int] = {}
        pin_ends: Dict[Tuple[int, int], List[Tuple[float, str, int]]] = {}

        for ts in thread_starts:
            pin = (ts[0] % m, ts[1] % n)
            pin_degree[pin] = pin_degree.get(pin, 0) + 1

        success = True
        for ti, ts in enumerate(thread_starts):
            path = _walk_one_thread(
                ti, ts, m, n, placed_segments, pin_degree, pin_ends,
                rng, max_steps_per_thread,
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

        if require_4_regular:
            non_four = {p: d for p, d in pin_degree.items() if d != 4}
            if non_four:
                if verbose:
                    print(f"  REJECTED: {len(non_four)} pins not deg-4")
                continue

        # Build a temporary graph dict and run is_lace_workable. Reject if
        # the global osculating circuit decomposition fails (e.g., (0,0)
        # circuits or otherwise non-workable globally).
        if _HAS_CHECKERS:
            tmp_result = {
                "n_cols": m, "n_rows": n,
                "starts": chosen_starts,
                "threads": [list(p) for p in thread_paths],
                "n_threads": len(thread_paths),
                "n_segments": len(placed_segments),
            }
            tmp_graph = to_lace_graph_dict(tmp_result)
            if not is_lace_workable(tmp_graph):
                if verbose:
                    print(f"  REJECTED: global lace_workability check failed")
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
# Output (same as lace_constructor)
# ---------------------------------------------------------------------------

def to_lace_graph_dict(result: dict, name: str = "constructed",
                        family: str = "bobbin") -> dict:
    n_cols = result["n_cols"]
    n_rows = result["n_rows"]

    visited = set()
    for path in result["threads"]:
        for (c, r) in path:
            visited.add((c % n_cols, r % n_rows))
    visited_pins = sorted(visited)
    pin_to_idx = {p: i for i, p in enumerate(visited_pins)}

    edges = []
    for path in result["threads"]:
        for i in range(len(path) - 1):
            sc, sr = path[i]
            dc, dr = path[i + 1]
            wrap_x = (dc // n_cols) - (sc // n_cols)
            wrap_y = (dr // n_rows) - (sr // n_rows)
            edges.append({
                "src": pin_to_idx[(sc % n_cols, sr % n_rows)],
                "dst": pin_to_idx[(dc % n_cols, dr % n_rows)],
                "wrap": [wrap_x, wrap_y],
                "polyline": [[sc, sr], [dc, dr]],
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


def render_construction(result: dict, output_path: str,
                          n_repeats: int = 3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = result["n_cols"]
    n_rows = result["n_rows"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for kx in range(-1, n_repeats + 1):
        for ky in range(-1, n_repeats + 1):
            for r in range(n_rows):
                for c in range(n_cols):
                    x = c + kx * n_cols
                    y = r + ky * n_rows
                    ax.plot(x, y, 'o', color='lightgray', markersize=4,
                             zorder=1)

    cmap = plt.get_cmap('tab10')
    for ti, path in enumerate(result["threads"]):
        color = cmap(ti % 10)
        for kx in range(-1, n_repeats + 1):
            for ky in range(-1, n_repeats + 1):
                xs = [p[0] + kx * n_cols for p in path]
                ys = [p[1] + ky * n_rows for p in path]
                ax.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.8,
                         zorder=2)

    cell_x = [0, n_cols, n_cols, 0, 0]
    cell_y = [0, 0, n_rows, n_rows, 0]
    ax.plot(cell_x, cell_y, '--', color='crimson', linewidth=1.5, alpha=0.8,
             zorder=3)

    ax.set_aspect('equal')
    ax.set_xlim(-1, n_repeats * n_cols + 1)
    ax.set_ylim(-1, n_repeats * n_rows + 1)
    ax.set_title(f"Bobbin lace: {n_rows}x{n_cols} cell, "
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
        description="Bobbin lace constructive generator (with kiss-compatible "
                    "enforcement at every pin)")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--bobbins", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=100)
    ap.add_argument("--vert-wrap-limit", type=int, default=2)
    ap.add_argument("--no-require-4-regular", action="store_true")
    ap.add_argument("--check-workability", action="store_true")
    ap.add_argument("--output-dir", type=str, default=".")
    ap.add_argument("--name", type=str, default="bobbin")
    ap.add_argument("--starts", type=str, default=None)
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

    print(f"Constructing bobbin lace: {args.n}x{args.m} cell, "
          f"{args.bobbins} starts ({2 * args.bobbins} threads), "
          f"seed={args.seed}, vert_wrap_limit={args.vert_wrap_limit}...")

    result = construct_bobbin_lace(
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
                print("Valid bobbin-lace ground!")


if __name__ == "__main__":
    main()
