# Aux-Mat Atlas — Project Status (May 4, 2026)

## What this is

A periodic graph metamaterials catalog with mechanical, beam, phonon,
and humidity physics for 360 textile-derived ground patterns.

Built on top of Veronika Irvine's TesseLace bobbin lace catalog (321
grounds, CC-BY 4.0, http://d-bl.github.io/tesselace-to-gf/) plus our
own constructive generator that produces additional bobbin lace ground
topologies satisfying Irvine's strict mathematical criterion.

The visualizer is a static-served webpage at `docs/index.html` showing
catalog grid, scatter plot of mechanical vs. humidity response, and
per-ground modal with physics tabs.

## Identity

- Repo: `/Users/richardtaylor/Dropbox/Research/Aux-Mat/Aux_mat_repo/`
- Owner: Richard Taylor (Independent), r_taylor@outlook.com
- Package: `auxetic-lace-atlas` v0.1.0
- Visible product name: "Aux-Mat Atlas"

## Status as of May 4, 2026 (end of day)

**Atlas state:** 360 grounds total
  - 321 from Irvine catalog
  - 39 from constructive generator (`bobbin_lace_constructor.py`),
    sweep at 3×3 b=2 with 100 seeds × 10000 attempts
  - All 360 are 4-regular, 2-in-2-out, planar, lace-workable

**Working tree:** clean (everything committed)

**Per-ground taxonomy (NEW — replaces old scheme):**
  - `family` — sorted set of face sizes in the planar embedding,
    joined with `_`. E.g. `3_6` (kagome), `3_4_5_7`, `3_4_6_8`.
    Computed from the graph by `compute_family.family_label()`.
    This is uniform across Irvine and Taylor grounds — same algorithm,
    same labels.
  - `traditional_name` — old non-numeric family if any. 4 cloth
    grounds → `"cloth"`, 39 Taylor grounds → `"taylor_bobbin"`. Most
    grounds: `null`.
  - `source` — `"irvine"` (321) or `"taylor"` (39). Drives marker
    shape in scatter plot and Provenance column in tables.

**Result of family migration:** all 39 Taylor grounds land in
pre-existing Irvine face-set families. No new face-set families
discovered, but new graph topologies within known families:
  - 19 grounds in `3_4_5_7` (Irvine had 30)
  - 7 in `3_4_6` (Irvine had 30)
  - 5 in `3_4_5_6` (Irvine had 30)
  - 4 in `3_4_5` (Irvine had 60)
  - 2 in `3_5` (Irvine had 12)
  - 1 in `3_4_6_8` (Irvine had 10)
  - 1 in `3_4_7` (Irvine had 19)

## Recent commits (chronological)

- `8b84c17`: bobbin lace constructor + periodic-copy crossing fix
- `31e5643`: integrate 39 new bobbin lace grounds (taylor_bobbin family)
- **Today's session:**
  - `3138cb5`: migrate atlas to face-set family labels + provenance
  - `4310cc8`: visualizer — provenance column, zero reference lines,
    source field
  - `4c214d3`: add face-set family computation and migration tooling
    (post-hoc)
  - `f18f720`: relocate Taylor ground thumbnails to face-set family
    folders
  - `b48dec9` (HEAD): catalog search — include source and
    traditional_name

## Architecture quick reference

### Atlas schema

`docs/atlas.json` has FOUR top-level keys:
- `metadata`: build-time params, n_grounds, attribution, etc.
- `summary`: lightweight per-ground entries used by the catalog grid
  and top-N table
- `grounds`: full per-ground records (physics, polylines, etc.)
- `failures`: any builds that crashed

CRITICAL: the visualizer uses `summary[]` for the catalog grid and
`metadata.n_grounds` for the count display. Updates to `grounds[]`
alone do NOT show up in the visualizer until `summary[]` and
`metadata` are rebuilt. See "How to update summary" below.

Each summary entry: `idx, name, family, source, traditional_name,
n_rows, n_cols, n_vertices, spring_default_nu_min,
spring_default_nu_max, spring_default_classification,
beam_default_nu_min, beam_default_nu_max,
beam_default_classification`.

(`source` and `traditional_name` were added in commit `3138cb5`; older
schema docs may not list them.)

### Per-ground record fields

Each entry in `grounds[]` has:
- `name` — e.g. `2x4_86`, `V9_3x3_007`
- `family` — face-set label, computed from embedding
- `source` — `"irvine"` or `"taylor"`
- `traditional_name` — string or null
- `n_rows`, `n_cols`, `n_vertices`, `n_edges`
- `vertices` — list of `[col, row]` integer tuples (y-down)
- `edges` — list of `{src, dst, wrap, polyline}` directed arcs
- `lattice`, `cell_area`
- `thumbnails` — paths under `docs/thumbnails/<family>/<name>/`
- `spring`, `beam`, `phonon`, `humidity` — physics blocks
- `graph_canonical`, `lace_canonical` — D4-canonicalized fingerprints
- `manufacturability` — `is_planar`, `is_printable`,
  `is_lace_workable`, `is_2in2out`
- `provenance` — build-time metadata (seed, params, code commit)

### Edge polyline frame convention

Polylines are 2-tuples in **source-anchored absolute lattice
coordinates**: `polyline[0]` equals the src vertex's coordinate,
`polyline[-1]` equals dst's "extended" position
(= dst coord + wrap × (n_cols, n_rows)). This is documented in
`parse_to_graph.py` and used by `compute_family.py`.

### Key code locations

| File | Purpose |
|------|---------|
| `src/auxetic_lace/canonicalize.py` | `graph_canonical(g)`, `lace_canonical(g)` — D4-canonicalized fingerprints. NOT in `canonical/` submodule. |
| `src/auxetic_lace/canonical/` | brute-force enumerator helpers. Has known limitations vs Irvine 5.2 — do not use for primary enumeration. |
| `src/auxetic_lace/parse_to_graph.py` | `LaceGraph` dataclass, `Edge` dataclass (frozen, fields: src_idx, dst_idx, wrap, polyline). `parse_file(path, name=, family=)`. Polylines are source-anchored lattice coords (see above). |
| `src/auxetic_lace/compute_family.py` | **NEW** `family_label(graph)` — sorted face-set label. `trace_faces(graph)`, `build_rotation_system(graph)`, `assign_traditional_name(name, old_family)`. Verified to agree with all 321 Irvine labels. |
| `src/auxetic_lace/build_atlas.py` | `build_ground_record(graph, name, family, theta_grid=None, thumbnail_dir=None)` — full per-ground physics + record |
| `src/auxetic_lace/lace_workability.py` | `is_lace_workable`, `_check_2_in_2_out`, `_build_vertex_rotation`, `_check_rotationally_consecutive`, `trace_osculating_circuits` |
| `src/auxetic_lace/planarity.py` | `is_planar` (polyline-aware), `_segments_cross_off_pin`, `_orient` |
| `src/auxetic_lace/manufacturability.py` | `is_2in2out` |
| `src/auxetic_lace/render_lace_view.py` | `render_ground_lace_views_split(graph, thread_path, n_tiles=5)` |
| `src/auxetic_lace/build_thumbnails.py` | only `thumbnail_paths` and `main` exposed (CLI: --catalog, --output-dir, --limit, --filter-family, --skip-lace, --skip-deformed, --skip-dispersion, --n-tiles, --force, --verbose) |
| `src/auxetic_lace/lace_constructor.py` | basic constructive generator (with periodic-copy crossing fix) |
| `src/auxetic_lace/bobbin_lace_constructor.py` | bobbin lace generator with kiss-compatible enforcement |
| `docs/index.html` | visualizer (single-file, static-served) |

### One-time migration scripts (kept as repo-root documentation)

| Script | Purpose |
|---|---|
| `verify_family_against_atlas.py` | Audit: compute face-set family for every ground in atlas.json; report disagreements with stored value. Used pre-migration to confirm algorithm correctness against Irvine ground truth (321/321 matched). |
| `integrate_face_set_family.py` | Migration: rewrites `family` field, adds `traditional_name` and `source`, rebuilds `summary[]`. Run with `--apply` to commit, dry-run otherwise. |
| `fix_source_field.py` | Idempotent fix-up: re-derives `source` from name pattern (V-prefix = taylor). Needed because the original `derive_source()` keyed off the soon-to-be-overwritten family field; running it on already-migrated atlas left source incorrect. |

### Bobbin lace ground definition (formal)

A bobbin lace ground is a 4-regular periodic 2-in-2-out planar graph
whose osculating-circuit decomposition (determined by kiss-compatible
vertex rotations) partitions the directed edges into closed circuits,
each wrapping the unit cell exactly once around the thread axis and
zero times around the perpendicular axis.

Specifically:
1. 4-regular (undirected, equivalently 2-in-2-out as directed)
2. Periodic in 2D
3. Vertex rotation rule: cyclic role pattern at each vertex must be
   `[in, in, out, out]` (some rotation), not transverse `[in, out, in, out]`
4. Edge partition: every directed edge in exactly one osculating circuit
5. Wrap rule: every circuit wraps `(0, ±1)` (or equivalently `(±1, 0)`
   in transposed convention) — once around one axis, zero around the other
6. Planar in the cell-coord embedding (no off-pin crossings between
   straight edges)

### Family taxonomy (face-set labels)

`family` is the **sorted set of distinct face sizes** in the planar
embedding, joined with `_`. Examples:
  - `4` — only square faces (square tiling, includes the cloth grounds)
  - `3_6` — triangular and hexagonal faces (kagome / Point de Paris)
  - `3_4_6` — triangular, square, hexagonal
  - `3_4_5_7` — triangular, square, pentagonal, heptagonal

Computed by `compute_family.family_label(graph)`. Algorithm:
1. Build rotation system at each vertex (cyclic order of incident
   darts sorted by departure angle).
2. Trace faces: from each unvisited dart, walk by taking the next
   dart in the rotation after the reverse of the arriving dart.
3. Each closed walk is a face; record its length.
4. Sort the distinct lengths and join with `_`.

Verified against all 321 Irvine grounds (100% agreement) before
migration.

### Naming convention

For taylor grounds: `V<n_vertices>_<n_rows>x<n_cols>_<serial>`
where serial is sequential within the (n_v, n_rows, n_cols) signature.
Example: `V9_3x3_001`, `V9_3x3_002`, ...

This naming is also the canonical signal of provenance: any name
matching `^V\d+_\d+x\d+_\d+$` is a Taylor ground (used by
`fix_source_field.py`).

### Thumbnail layout

Thumbnails live under `docs/thumbnails/<family>/<name>/`:
- `thread.png` — exists for all 360 grounds
- `deformed.png` — exists for 321 Irvine grounds, **NOT YET** for the
  39 Taylor grounds
- `dispersion.png` — same: Irvine yes, Taylor no

After the family migration, Taylor thumbnails were relocated from
`thumbnails/taylor_bobbin/<name>/` to `thumbnails/<face_set_family>/<name>/`
in commit `f18f720`. The visualizer builds the URL from
`s.family` directly.

## Server & visualizer notes

```bash
cd /Users/richardtaylor/Dropbox/Research/Aux-Mat/Aux_mat_repo/docs
python3 -m http.server 8000
```

Then http://localhost:8000/index.html. If browser shows stale data
that doesn't match disk:

1. First check `summary[]` and `metadata.n_grounds` — these are often
   the culprit, NOT browser cache. Run:
   ```bash
   python3 -c "import json; a=json.load(open('docs/atlas.json'));
   print(f'meta.n_grounds={a[\"metadata\"][\"n_grounds\"]}, '
         f'summary={len(a[\"summary\"])}, grounds={len(a[\"grounds\"])}')"
   ```
2. If summary/metadata are stale, rebuild from grounds (see "How to
   update summary" below).
3. Hard-refresh via Cmd+Shift+R only AFTER summary is correct.
4. Verify what server actually serves: `curl -s http://localhost:8000/atlas.json | md5`

### How to update summary[] from grounds[]

```python
import json
with open("docs/atlas.json") as f:
    a = json.load(f)

spring_default_idx = a["metadata"].get("spring_default_idx", 2)
beam_default_idx = a["metadata"].get("beam_default_idx", 1)

new_summary = []
for idx, g in enumerate(a["grounds"]):
    spring = g.get("spring", {}) or {}
    beam = g.get("beam", {}) or {}
    def safe(arr, i):
        return arr[i] if arr and i < len(arr) else None
    new_summary.append({
        "idx": idx,
        "name": g["name"],
        "family": g["family"],
        "source": g.get("source", "irvine"),
        "traditional_name": g.get("traditional_name"),
        "n_rows": g["n_rows"], "n_cols": g["n_cols"],
        "n_vertices": g["n_vertices"],
        "spring_default_nu_min": safe(spring.get("nu_min"), spring_default_idx),
        "spring_default_nu_max": safe(spring.get("nu_max"), spring_default_idx),
        "spring_default_classification":
            safe(spring.get("classification"), spring_default_idx),
        "beam_default_nu_min": safe(beam.get("nu_min"), beam_default_idx),
        "beam_default_nu_max": safe(beam.get("nu_max"), beam_default_idx),
        "beam_default_classification":
            safe(beam.get("classification"), beam_default_idx),
    })

a["summary"] = new_summary
a["metadata"]["n_grounds"] = len(a["grounds"])

with open("docs/atlas.json", "w") as f:
    json.dump(a, f)
```

(`integrate_face_set_family.py` does this with atomic write and
validation; prefer it for any future changes.)

### Scatter plot logic location

`docs/index.html`, function `renderScatterPlot()`, around line 131.
Filters to lace-workable grounds, plots `nu_min` (beam_default_idx)
on x-axis vs `eta_pore` (humidity) on y-axis.

- **Color:** by classification (homogeneous=red / directional=orange /
  non-auxetic=grey).
- **Marker shape:** circle for `source === "irvine"`, plus (`+`) for
  `source === "taylor"`. Logic at line ~308.
- **Reference lines:** dashed grey lines at literal `x=0` and `y=0`.
  These are the physical thresholds (auxetic ⇔ ν_min < 0,
  humidity-active ⇔ η_pore < 0). The four corner labels (`auxetic
  only`, `strong both`, `humidity-active only`, `neither`) refer to
  quadrants of this partition.

(Pre-May 4, the dashed lines were at the median of the data, which
shifted as the catalog grew. Switched to zero in commit `4310cc8`.)

### Top homogeneously-auxetic table

`docs/index.html`, around line 499–520. Columns: Family | Provenance |
Name | ν_min | ν_max | Class. The Provenance column reads `s.source`
and renders "Irvine" or "Taylor".

### Catalog search

`docs/index.html`, `renderCatalogGrid()`, search haystack around line
~563. Searches across `family`, `name`, `source`, and
`traditional_name`. So:
  - `taylor` matches the 39 Taylor grounds (via source or
    traditional_name)
  - `cloth` matches the 4 cloth grounds (via traditional_name)
  - `irvine` matches all 321 Irvine grounds (via source)
  - face-set codes (`3_6`, `3_4_6_8`) match by family
  - V-names match by name

## Pending tasks (priority order)

### Open: thumbnails for Taylor grounds
- ✅ thread.png exists for all 39 (now under their face-set family
  folders, not taylor_bobbin)
- ❌ **deformed.png not yet rendered** for Taylor grounds
- ❌ **dispersion.png not yet rendered** for Taylor grounds
- Modal tabs for these will 404 until rendered. `build_thumbnails.py`
  only exposes main(); options:
  (a) Run with `--filter-family taylor_bobbin` if that flag exists —
      but note that AFTER the family migration, taylor_bobbin is no
      longer a family value; Taylor grounds now belong to face-set
      families. The CLI flag may need updating, OR pass the new
      face-set families one at a time, OR filter by `source=="taylor"`
      via a new CLI option.
  (b) Adapt the existing thumbnail rendering code into a per-ground
      function callable directly.

### Open: status doc as committed artifact
- This file (or whatever it's named) should now live in the repo so
  that future updates flow through git. Decide on a path
  (suggest `docs/PROJECT_STATUS.md` or `STATUS.md` at root) and add
  to first-class tooling.

### Open: overnight sweep
Goal: explore larger cells and bobbin counts to grow Taylor catalog.
Conservative parameter ranges:
  - Cell sizes: 3×3 (already done), 3×4, 4×3, 4×4
  - bobbins: 2, 3
  - Many seeds per combination
  - 10000 attempts per seed

Constraints from earlier testing:
  - 4×4 b=3: 0 successes in earlier 6-seed test — needs much higher
    attempts at higher density
  - 3×3 b=3: 0 successes in earlier 6-seed test — same issue
  - 3×3 b=2: 65/100 success rate at 10000 attempts — productive

Implementation notes:
  - Save outputs to persistent storage (NOT /tmp, e.g.
    `~/Dropbox/Research/Aux-Mat/bobbin_sweep_<date>/`)
  - Use `integrate_bobbin_sweep.py` to add successful unique grounds
    into atlas after sweep. NOTE this script will need to be updated
    to assign `family` via face-set computation rather than the
    obsolete `family="taylor_bobbin"` literal, AND to set
    `source="taylor"` and `traditional_name="taylor_bobbin"`.
  - Resumable: skip already-tested seeds on restart
  - Per-seed timeout (5 min) to bound worst case
  - Naming: continue the `V<n>_<r>x<c>_<serial>` convention

### Future
- LaTeX paper text formalizing bobbin lace ground definition
  (drafted in transcripts; both prose and LaTeX-form versions exist)
- Scale up sweep to 5×5+ cells once thumbnails done
- Investigate stitch-based smarter construction (rather than random
  walks) for higher success rates at dense parameters
- Cross-cell-size dedup via canonical (already working — graph_canonical
  correctly identifies isomorphic patterns across cell sizes)

## Operational notes

- macOS Downloads behavior: re-downloading a file with same name
  produces "(1)" suffix. Workflow: `rm -f ~/Downloads/<file>.py`
  before downloading new version.
- macOS Terminal auto-link: pasting filenames containing underscores
  may turn `foo_bar.py` into `foo_[bar.py](http://bar.py)` and break
  commands. Workarounds: use wildcards (`foo*`), type the name by
  hand, or disable smart copy/paste in Terminal preferences.
- zsh quoting: heredocs with `python3 << 'PYEOF'` reliable;
  `python3 -c "..."` with `$"..."` mixing causes hangs. Multi-line
  `git commit -m "..."` messages also hang waiting for closing quote;
  prefer single-line messages or open the editor without `-m`.
- Server: `python3 -m http.server 8000` from `docs/` directory.
- **Browser cache vs server data**: when troubleshooting "page shows
  wrong data", always check the server output directly with curl first
  (`curl -s http://localhost:8000/atlas.json | md5`), THEN check
  browser cache. Most problems hit were summary[] being stale
  (server-side data), NOT browser caching.

## Sweep result archive

`~/Dropbox/Research/Aux-Mat/bobbin_sweep_2026_05_03/` has the 65
successful sweep outputs (each is a directory with graph.json and
lace.png). 42 are unique by canonical; 39 are new vs Irvine.
