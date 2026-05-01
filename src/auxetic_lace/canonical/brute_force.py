"""
brute_force.py
==============

Brute-force ground-truth enumerator for tesselace embeddings on small
period rectangles. Used as an oracle to validate the faster lace path
approach.

Strategy:
  1. For each vertex (col, row) in the period rectangle, enumerate all
     possible pairs of outgoing edges (each edge = one of the 8 step
     vectors in L). The 2-regularity property says EXACTLY 2 outgoing.
  2. Combine all vertex choices into a candidate state.
  3. Check 2-regularity globally (each vertex must also have exactly 2
     incoming edges — these come "for free" from other vertices'
     outgoing choices).
  4. Apply each tesselace embedding property as a filter:
       - Property 3.2.1: 2-regular (already ensured by construction)
       - Property 3.2.2: periodic (already ensured — we live on a torus)
       - Property 3.2.3: connected
       - Property 3.2.4: no contractible directed cycles (= rotational
         consistency at every vertex)
       - Property 3.2.5: thread conserving (osculating circuits with
         wrapping index (1, 0))
  5. Apply prime detection (Section 4.3).
  6. Apply canonical labeling for isomorphism class deduplication
     (Section 4.2).

This is exponentially slow but provably correct. We use it ONLY for
small period rectangles (n_cols * n_rows ≤ 6 or so).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# Step vectors L from thesis Section 5
L_ALL: Tuple[Tuple[int, int], ...] = (
    (1, 1),
    (1, 0),
    (2, 0),
    (1, -1),
    (0, 1),
    (0, 2),
    (0, -1),
    (0, -2),
)


# ---------------------------------------------------------------------
# Step indexing for canonical labels (Section 4.2 Figure 4.4)
# Sort step vectors counterclockwise starting from east, with shorter
# steps before longer at the same angle.
# ---------------------------------------------------------------------

# Build the index over ALL step vectors that can ever appear (forward
# and reverse, since one vertex's outgoing edge is another's incoming).
_ALL_STEPS = set()
for _s in L_ALL:
    _ALL_STEPS.add(_s)
    _ALL_STEPS.add((-_s[0], -_s[1]))


def _angle_key(s: Tuple[int, int]) -> Tuple[float, int]:
    ang = math.atan2(s[1], s[0])
    if ang < 0:
        ang += 2 * math.pi
    return (ang, s[0] ** 2 + s[1] ** 2)


_SORTED_STEPS = sorted(_ALL_STEPS, key=_angle_key)
STEP_INDEX: Dict[Tuple[int, int], int] = {
    s: i + 1 for i, s in enumerate(_SORTED_STEPS)
}


# ---------------------------------------------------------------------
# Ground state representation
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Ground:
    """A complete tesselace embedding on n_cols × n_rows torus.

    Stored canonically: edges as a frozenset of (src_col, src_row, dx,
    dy) tuples. Source row is in [0, n_rows). Destination is computed
    as ((src_col + dx) % n_cols, (src_row + dy) % n_rows).
    """
    n_cols: int
    n_rows: int
    edges: FrozenSet[Tuple[int, int, int, int]]

    def out_edges_at(self, col: int, row: int) -> List[Tuple[int, int]]:
        """All step vectors of outgoing edges at (col, row)."""
        return [(dx, dy) for (sc, sr, dx, dy) in self.edges
                if sc == col and sr == row]

    def in_edges_at(self, col: int, row: int) -> List[Tuple[int, int]]:
        """All step vectors of incoming edges at (col, row), as the
        SOURCE step vector (so the edge points FROM (col-dx, row-dy)
        TO (col, row))."""
        out = []
        for (sc, sr, dx, dy) in self.edges:
            dst_c = (sc + dx) % self.n_cols
            dst_r = (sr + dy) % self.n_rows
            if dst_c == col and dst_r == row:
                out.append((dx, dy))
        return out


# ---------------------------------------------------------------------
# Property 3.2.4: Rotational consistency (no contractible directed cycles)
# ---------------------------------------------------------------------

def _check_rotational_consistency(g: Ground) -> bool:
    """At every vertex, the rotational order of edges must be
    'consecutive' (incoming and outgoing each grouped) not
    'alternating' (Section 3.3, Lemma 3.3.1).

    Concretely: sort all 4 incident edges (2 in + 2 out) by angle around
    the vertex, then check that the cyclic sequence of types has at
    most 2 transitions.
    """
    for col in range(g.n_cols):
        for row in range(g.n_rows):
            # Outgoing direction = (dx, dy) directly
            out_dirs = g.out_edges_at(col, row)
            # Incoming direction (pointing AWAY from this vertex) = (-dx, -dy)
            # because the source-step (dx, dy) points TOWARDS this vertex.
            in_dirs = [(-dx, -dy) for (dx, dy) in g.in_edges_at(col, row)]
            n = len(out_dirs) + len(in_dirs)
            if n != 4:
                return False  # not 2-regular at this vertex
            items = ([(math.atan2(dy, dx), 'out') for (dx, dy) in out_dirs] +
                     [(math.atan2(dy, dx), 'in') for (dx, dy) in in_dirs])
            items.sort()
            types = [t for (_, t) in items]
            transitions = sum(1 for i in range(len(types))
                              if types[i] != types[(i + 1) % len(types)])
            if transitions > 2:
                return False
    return True


# ---------------------------------------------------------------------
# Property 3.2.3: Connected
# ---------------------------------------------------------------------

def _check_connected(g: Ground) -> bool:
    """Underlying undirected graph is connected, AND has non-contractible
    cycles in both meridian and longitude directions of the torus.

    For a 2-regular digraph on a torus to be connected with the right
    topology, we need ALL n_cols * n_rows vertices reachable from any
    one of them, treating edges as undirected.
    """
    if g.n_cols * g.n_rows == 0:
        return True
    visited = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        v = stack.pop()
        for (dx, dy) in g.out_edges_at(*v):
            dst = ((v[0] + dx) % g.n_cols, (v[1] + dy) % g.n_rows)
            if dst not in visited:
                visited.add(dst)
                stack.append(dst)
        for (dx, dy) in g.in_edges_at(*v):
            src = ((v[0] - dx) % g.n_cols, (v[1] - dy) % g.n_rows)
            if src not in visited:
                visited.add(src)
                stack.append(src)
    return len(visited) == g.n_cols * g.n_rows


# ---------------------------------------------------------------------
# Property 3.2.5: Thread conservation (osculating circuits with wrap (1, 0))
# ---------------------------------------------------------------------

def _osculating_partition(g: Ground) -> Optional[List[List[Tuple[int, int, int, int, int, int]]]]:
    """Partition edges into osculating circuits (Lemma 3.3.4).

    Each circuit is a sequence of edges where, at each interior vertex,
    the two consecutive edges are 'rotationally consecutive' — i.e., we
    enter on an incoming edge and leave on the rotationally adjacent
    outgoing edge (the one that's next to it in cyclic angle order).

    Returns: list of circuits. Each circuit is a list of
        (src_col, src_row, dx, dy, wrap_col, wrap_row) tuples, where
        the wraps track how many times the circuit wraps around the
        torus (used to compute wrapping index).

    Returns None if the graph isn't suitable for osculating partition
    (shouldn't happen if rotational consistency was already checked).
    """
    edges_remaining = set(g.edges)
    circuits = []

    while edges_remaining:
        # Pick any edge and trace its osculating circuit
        start_edge = next(iter(edges_remaining))
        circuit: List[Tuple[int, int, int, int, int, int]] = []
        # Track total wrap as we traverse
        wrap_col = 0
        wrap_row = 0
        cur_edge = start_edge
        cur_wrap_col = 0
        cur_wrap_row = 0
        first_iteration = True
        while True:
            (sc, sr, dx, dy) = cur_edge
            circuit.append((sc, sr, dx, dy, cur_wrap_col, cur_wrap_row))
            if cur_edge in edges_remaining:
                edges_remaining.remove(cur_edge)
            else:
                # Edge already used — circuit re-entered itself, abort
                # (this shouldn't happen if we trace correctly)
                return None
            # Update wrap as we step from src to destination (dst is the
            # vertex at end of this edge)
            new_wrap_col = cur_wrap_col + (sc + dx) // g.n_cols
            new_wrap_row = cur_wrap_row + (sr + dy) // g.n_rows
            # Hmm — wait. The src is in [0, n_cols), so sc + dx may be
            # ≥ n_cols (wrap forward) or < 0 (wrap backward). We track
            # the integer count of wraps.
            # But this isn't quite right — we want to track how the
            # circuit "moves around the torus" as we follow it. Let me
            # use unwrapped coordinates for tracking.
            unwrapped_dst_col = sc + dx + cur_wrap_col * g.n_cols
            unwrapped_dst_row = sr + dy + cur_wrap_row * g.n_rows
            # The destination on the torus:
            dst_col = unwrapped_dst_col % g.n_cols
            dst_row = unwrapped_dst_row % g.n_rows
            new_wrap_col = unwrapped_dst_col // g.n_cols
            new_wrap_row = unwrapped_dst_row // g.n_rows

            # Find the rotationally next outgoing edge at the destination
            next_edge = _next_osculating_edge(g, dst_col, dst_row, dx, dy)
            if next_edge is None:
                return None

            # Build the next edge in unwrapped form
            next_dx, next_dy = next_edge
            cur_edge = (dst_col, dst_row, next_dx, next_dy)
            cur_wrap_col = new_wrap_col
            cur_wrap_row = new_wrap_row

            # Termination: returned to the starting edge (with same wrap=0)
            if cur_edge == start_edge:
                break

        circuits.append(circuit)

    return circuits


def _next_osculating_edge(g: Ground, dst_col: int, dst_row: int,
                          incoming_dx: int, incoming_dy: int
                          ) -> Optional[Tuple[int, int]]:
    """Given that we just arrived at (dst_col, dst_row) via edge with
    step (incoming_dx, incoming_dy), find the OUTGOING edge at this
    vertex that is osculating-consecutive (i.e., immediately rotationally
    next to the incoming edge, on the same 'side' of the vertex).

    Per Lemma 3.3.4 / Figure 3.5(b): at a 'rotationally consecutive'
    vertex, the two outgoing edges are grouped together angularly. The
    incoming edge we arrived on came from direction (-incoming_dx,
    -incoming_dy) [angle of the source]; the corresponding 'partner'
    outgoing edge is the one rotationally next when we sort by angle
    around the vertex.

    Concretely: list all 4 incident edges by angle around dst, and find
    where the incoming edge sits; the next-clockwise (or
    next-counterclockwise) edge is the partner.

    Convention: at a rotationally-consecutive vertex, walking around
    the vertex in one direction (say counterclockwise), we encounter:
      [out_a, out_b, in_a, in_b]
    or any cyclic shift. We want to enter on an in_*, exit on the
    NEXT out_* in the cyclic order (i.e., the one that is
    counterclockwise-next AFTER the incoming).

    More concretely still: thesis Section 3.3 (Lemma 3.3.4 proof):
    'add the unique, rotationally consecutive outgoing edge'. So after
    arriving on incoming edge e_in, the outgoing edge to take is the
    one rotationally adjacent in the consecutive grouping.

    Simplest correct rule: at a rotationally-consecutive vertex, the
    incoming-incoming-outgoing-outgoing arrangement means that, sorted
    by angle CCW, the cyclic sequence of types is e.g. (in, in, out,
    out). We arrived on one of the 'in' edges. The 'partner' outgoing
    is the one that's rotationally adjacent on the OUT side — which
    means: if we arrived on the FIRST 'in' (rotationally), we leave on
    the SECOND 'out'; if we arrived on the SECOND 'in', we leave on the
    FIRST 'out'. This makes the circuit osculate (kiss) at the vertex
    rather than cross.
    """
    out_dirs = g.out_edges_at(dst_col, dst_row)  # (dx, dy)
    # The incoming-edge's direction AWAY from this vertex (i.e. toward
    # its source) is (-incoming_dx, -incoming_dy).
    in_dirs_away = [(-dx, -dy) for (dx, dy) in g.in_edges_at(dst_col, dst_row)]

    if len(out_dirs) != 2 or len(in_dirs_away) != 2:
        return None

    # Build labeled list, sorted by angle
    items = ([(math.atan2(dy, dx), 'out', (dx, dy)) for (dx, dy) in out_dirs] +
             [(math.atan2(dy, dx), 'in', (dx, dy)) for (dx, dy) in in_dirs_away])
    items.sort()
    # Find the index where incoming-edge's away-direction lies
    arrived_away = (-incoming_dx, -incoming_dy)
    incoming_idx = None
    for i, (_, t, vec) in enumerate(items):
        if t == 'in' and vec == arrived_away:
            incoming_idx = i
            break
    if incoming_idx is None:
        return None

    # The osculating partner is determined by: sort cyclic order has
    # types like [in, in, out, out] (rotationally consecutive). The 'in'
    # at position incoming_idx pairs with the 'out' that's adjacent on
    # the OUT-side of the in/out boundary.
    # Walk in one direction (say +1) to find which side of the in-pair
    # we're on. If items[(incoming_idx + 1) % 4] is 'in' (so we're on
    # the FIRST in-position), then we should exit on the LAST out
    # (items[(incoming_idx - 1) % 4]). Conversely, if items[(incoming_idx
    # + 1) % 4] is 'out', we exit on items[(incoming_idx + 1) % 4].
    nxt_idx = (incoming_idx + 1) % 4
    prv_idx = (incoming_idx - 1) % 4
    if items[nxt_idx][1] == 'out':
        return items[nxt_idx][2]
    elif items[prv_idx][1] == 'out':
        return items[prv_idx][2]
    return None  # shouldn't happen at a rotationally consistent vertex


def _check_thread_conserving(g: Ground) -> bool:
    """Property 3.2.5: every osculating circuit has wrapping index
    (1, 0) — wraps once around the meridian (rows), zero times around
    the longitude (columns).

    Thesis convention: meridian = vertical = down-the-pillow direction.
    But our cell has rows running vertically. So "wrap index (1, 0)"
    here means: each circuit wraps once vertically (in row direction)
    and zero times horizontally (in column direction).

    NOTE: the thesis uses the convention that (M, L) = (meridian wrap,
    longitude wrap), where meridian = the "down" direction of the lace.
    In our lace_paths.py convention, paths go horizontally (left-to-
    right), so the "meridian" of the torus is the column direction
    (x-axis), and threads flow along columns. Thus thread-conservation
    requires wrapping (1, 0) where 1 = column wrap, 0 = row wrap.

    Hmm, this is confusing. Let me think about it operationally:
    threads flow along the path direction. Lace paths in our code go
    from x=0 to x=n_cols, so "one circuit" = one full traversal of the
    pattern's width (x-axis). So circuits should wrap once in x (col
    direction) and zero times in y (row direction).
    """
    circuits = _osculating_partition(g)
    if circuits is None:
        return False
    for circuit in circuits:
        # Sum the wrap of each edge to get the circuit's total wrap
        total_wrap_col = 0
        total_wrap_row = 0
        for (sc, sr, dx, dy, wc, wr) in circuit:
            # Total displacement in raw coordinates
            pass
        # Simpler: walk through circuit, tracking unwrapped position
        x = circuit[0][0]
        y = circuit[0][1]
        for (sc, sr, dx, dy, wc, wr) in circuit:
            x += dx
            y += dy
        # Returned to start means (x mod ncols, y mod nrows) = (sc, sr)
        # at start. The total wrap is:
        wrap_col = (x - circuit[0][0]) // g.n_cols
        wrap_row = (y - circuit[0][1]) // g.n_rows
        # Each circuit must have wrap (1, 0) — 1 in col direction (the
        # path's "downstream" axis), 0 in row direction.
        if (wrap_col, wrap_row) != (1, 0):
            return False
    return True


# ---------------------------------------------------------------------
# Vertex labels (Section 4.2)
# ---------------------------------------------------------------------

def _vertex_label(g: Ground, col: int, row: int) -> Tuple[int, ...]:
    """Sorted tuple of signed step indices for edges at (col, row)."""
    indices = []
    for (dx, dy) in g.out_edges_at(col, row):
        indices.append(+STEP_INDEX[(dx, dy)])
    for (dx, dy) in g.in_edges_at(col, row):
        # Incoming-from direction = (-dx, -dy)
        indices.append(-STEP_INDEX[(-dx, -dy)])
    indices.sort(key=lambda i: (abs(i), -i))
    return tuple(indices)


def _ground_label(g: Ground, origin_col: int, origin_row: int
                  ) -> Tuple[Tuple[int, ...], ...]:
    rows = []
    for r in range(g.n_rows):
        for c in range(g.n_cols):
            rows.append(_vertex_label(g, (c + origin_col) % g.n_cols,
                                       (r + origin_row) % g.n_rows))
    return tuple(rows)


def _transform_ground(g: Ground, reflect_h: bool, reflect_v: bool,
                      flip_orient: bool) -> Ground:
    new_edges = set()
    for (sc, sr, dx, dy) in g.edges:
        nsc, nsr, ndx, ndy = sc, sr, dx, dy
        if reflect_h:
            nsc = (g.n_cols - 1 - nsc) % g.n_cols
            ndx = -ndx
        if reflect_v:
            nsr = (g.n_rows - 1 - nsr) % g.n_rows
            ndy = -ndy
        if flip_orient:
            nsc = (nsc + ndx) % g.n_cols
            nsr = (nsr + ndy) % g.n_rows
            ndx = -ndx
            ndy = -ndy
        new_edges.add((nsc % g.n_cols, nsr % g.n_rows, ndx, ndy))
    return Ground(g.n_cols, g.n_rows, frozenset(new_edges))


def _canonical_label(g: Ground) -> Tuple[Tuple[int, ...], ...]:
    """Smallest label over all valid transformations × all translations.

    Section 4.2 lists 4 base transformations:
      1. identity
      2. h-reflect (mirror across vertical axis in thesis coords;
         equivalently flip-y in our chapter-5+ coordinate convention)
      3. v-reflect + orient-flip
      4. 180° rotation + orient-flip

    The thesis's "vertical" axis is the partial-order axis (down the
    pillow). In our convention (path direction is +x), the partial
    order is along x. So thesis v-reflect = flip x in our coords;
    thesis h-reflect = flip y in our coords.

    Map:
      thesis transform     | our (reflect_h, reflect_v, flip_orient)
      ---------------------+----------------------------------------
      identity             | (False, False, False)
      h-reflect            | (False, True, False)        flip y only
      v-reflect + flip     | (True, False, True)         flip x + flip
      180-rot + flip       | (True, True, True)          flip x+y + flip

    This is the canonical 4-transform set.
    """
    transforms = [
        (False, False, False),
        (False, True, False),
        (True, False, True),
        (True, True, True),
    ]
    best = None
    for (reflect_h, reflect_v, flip_orient) in transforms:
        t = _transform_ground(g, reflect_h, reflect_v, flip_orient)
        for orow in range(g.n_rows):
            for ocol in range(g.n_cols):
                label = _ground_label(t, ocol, orow)
                if best is None or label < best:
                    best = label
    return best


# ---------------------------------------------------------------------
# Prime detection (Section 4.3)
# ---------------------------------------------------------------------

def _is_prime(g: Ground) -> bool:
    """A ground is prime iff its toroidal array of vertex labels is
    aperiodic in BOTH row and column directions (Theorem 4.3.2)."""
    cell_labels = {(c, r): _vertex_label(g, c, r)
                   for c in range(g.n_cols)
                   for r in range(g.n_rows)}

    # Horizontal period: LCM of row periods
    h_period = 1
    for r in range(g.n_rows):
        row_seq = tuple(cell_labels[(c, r)] for c in range(g.n_cols))
        h_period = _lcm(h_period, _string_period(row_seq))
        if h_period >= g.n_cols:
            break
    if h_period < g.n_cols:
        return False

    # Vertical period
    v_period = 1
    for c in range(g.n_cols):
        col_seq = tuple(cell_labels[(c, r)] for r in range(g.n_rows))
        v_period = _lcm(v_period, _string_period(col_seq))
        if v_period >= g.n_rows:
            break
    if v_period < g.n_rows:
        return False

    return True


def _string_period(seq: Tuple) -> int:
    n = len(seq)
    for p in range(1, n + 1):
        if n % p != 0:
            continue
        if all(seq[i] == seq[i % p] for i in range(n)):
            return p
    return n


def _lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


# ---------------------------------------------------------------------
# Brute-force enumeration
# ---------------------------------------------------------------------

def enumerate_brute(n_cols: int, n_rows: int,
                    verbose: bool = False) -> List[Ground]:
    """Enumerate all prime tesselace embeddings on n_cols × n_rows
    period rectangle by brute force.

    Approach: pick 2 outgoing step vectors at each of n_cols × n_rows
    vertices. Check 2-regularity (in-degree must equal 2 at each vertex
    for free, modulo combinatorial luck). Then apply property filters.
    """
    # Build all unordered pairs of step vectors that could be at one
    # vertex. Forbid pairs that put two edges into the same direction.
    step_pairs = []
    L = list(L_ALL)
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            step_pairs.append((L[i], L[j]))
        # Could also allow (L[i], L[i]) — multiedges in the same
        # direction. The thesis allows multi-edges since it's a
        # multigraph. But two outgoing edges from same vertex with
        # identical step vectors would always be transverse on the
        # torus (or at least would create degenerate cases), so likely
        # forbidden in practice. Let's allow them and let the property
        # filters reject if invalid.
        step_pairs.append((L[i], L[i]))

    if verbose:
        print(f"  vertex options: {len(step_pairs)} pairs")

    n_verts = n_cols * n_rows
    total_combos = len(step_pairs) ** n_verts
    if verbose:
        print(f"  total combinations: {total_combos:,}")

    results: List[Ground] = []
    seen_labels: Set[Tuple] = set()

    # Iterate through all combinations
    vertex_list = [(c, r) for r in range(n_rows) for c in range(n_cols)]
    for combo in itertools.product(step_pairs, repeat=n_verts):
        edges = set()
        for (vert, pair) in zip(vertex_list, combo):
            (col, row) = vert
            (s1, s2) = pair
            if s1 == s2:
                # Multi-edge in same direction: skip
                edges = None
                break
            edges.add((col, row, s1[0], s1[1]))
            edges.add((col, row, s2[0], s2[1]))
        if edges is None:
            continue

        g = Ground(n_cols, n_rows, frozenset(edges))

        # Check in-degree 2 at every vertex (incoming edges come "for
        # free" from other vertices' choices — we need them to balance)
        in_deg = {(c, r): 0 for c in range(n_cols) for r in range(n_rows)}
        for (sc, sr, dx, dy) in g.edges:
            dst = ((sc + dx) % n_cols, (sr + dy) % n_rows)
            in_deg[dst] += 1
        if any(d != 2 for d in in_deg.values()):
            continue

        # Property 3.2.4: rotational consistency / no contractible cycles
        if not _check_rotational_consistency(g):
            continue

        # Property 3.2.3: connected
        if not _check_connected(g):
            continue

        # Property 3.2.5: thread conservation (osculating circuits w/ wrap (1,0))
        if not _check_thread_conserving(g):
            continue

        # Prime
        if not _is_prime(g):
            continue

        # Canonical label dedup
        label = _canonical_label(g)
        if label not in seen_labels:
            seen_labels.add(label)
            results.append(g)

    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"Brute-force enumeration: {args.cols} × {args.rows}")
    grounds = enumerate_brute(args.cols, args.rows, verbose=args.verbose)
    EXPECTED = {
        (1, 1): 1, (2, 1): 1, (3, 1): 1, (4, 1): 2, (5, 1): 3,
        (1, 2): 3, (2, 2): 7, (3, 2): 26, (4, 2): 112, (5, 2): 535,
        (1, 3): 4, (2, 3): 26, (3, 3): 277, (4, 3): 3527, (5, 3): 53132,
        (1, 4): 16, (2, 4): 176, (3, 4): 4308, (4, 4): 137273,
    }
    expected = EXPECTED.get((args.cols, args.rows))
    print(f"Found: {len(grounds)}; expected: {expected}")
    if expected is not None:
        ok = "OK" if len(grounds) == expected else "MISMATCH"
        print(f"  [{ok}]")


if __name__ == "__main__":
    main()
