# Contributing to the Auxetic Lace Atlas

This atlas is designed to grow through community contributions. If you
have a periodic 2D directed graph that defines an interesting structure
(a known auxetic mechanism, a novel topology, or a candidate worth
testing), you can submit it. The pipeline validates structure, computes
mechanics under both the spring and beam models, ranks against the
existing catalog, and either lands locally or merges via pull request.

## What can I submit?

Two flavors of submission are accepted:

1. **Tesselace-compliant graphs** — bobbin lace ground patterns
   satisfying Irvine's 5 mathematical properties (2-regular, periodic,
   connected, partial-order, thread-conserving). Set
   `tesselace_compliant: true` in the JSON and run validation with
   `--check-tesselace`.

2. **General periodic frameworks** — any 2-regular doubly-periodic
   directed graph. Includes structures like rotating squares, re-entrant
   honeycombs, and chiral lattices that don't satisfy thread
   conservation. Set `tesselace_compliant: false` (the default).

The atlas distinguishes the two via the `family` prefix:
- `tesselace/{hole_shapes}` for tesselace-compliant entries
- `user/{anything}` for general submissions

## Submission file format

A submission is a single JSON file. Minimum example
(`square_lattice.json`):

```json
{
  "name": "square_1x1",
  "family": "user/square",
  "n_rows": 1,
  "n_cols": 1,
  "vertices": [[0, 0]],
  "edges": [
    {"src": 0, "dst": 0, "wrap": [1, 0]},
    {"src": 0, "dst": 0, "wrap": [0, 1]}
  ]
}
```

Full schema in `contrib/submission_schema.py`. See
`submission_examples/` for working examples
(square, reentrant, triangular).

### Field meanings

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Short identifier, alphanumeric + `_-` only. Unique within `family`. |
| `family` | string | no | Family name. Default `user/contributed`. |
| `tesselace_compliant` | bool | no | Self-asserted. Default `false`. |
| `n_rows`, `n_cols` | int | yes | Period parallelogram dimensions. |
| `vertices` | `[[col, row], ...]` | yes | Integer unit-cell coordinates in `[0, n_cols) × [0, n_rows)`. |
| `edges` | `[{"src": i, "dst": j, "wrap": [wc, wr]}, ...]` | yes | Directed edges with periodic-image wrap. |
| `lattice_vectors` | 2x2 array | no | Default `[[1,0], [0,1]]`. Set for non-orthogonal lattices. |
| `provenance` | object | no | Author, description, DOI, license. |

### Edge wrap convention

An edge `{src: i, dst: j, wrap: [wc, wr]}` represents a directed bar
from `vertices[i]` to the periodic image of `vertices[j]` shifted by
`(wc, wr)` whole unit cells. So:
- `wrap: [0, 0]` is an edge entirely within one unit cell tile.
- `wrap: [1, 0]` reaches into the unit cell to the right.
- `wrap: [-1, 1]` reaches into the unit cell up and to the left.

Self-loops are valid as long as their `wrap` is nonzero (a self-loop
with `wrap: [0, 0]` would have zero length and is rejected).

## Local workflow

### Step 1: Validate

```bash
python3 contrib/validate_submission.py my_pattern.json
```

Checks: schema, 2-regular invariant, nonzero edge lengths,
connectedness. Add `--check-tesselace` to also verify the partial-order
and thread-conservation properties.

Exit codes:
- `0` = passed
- `1` = structural errors
- `2` = passed structural but failed tesselace checks (only with
  `--strict` or `tesselace_compliant: true`)

### Step 2: Compute mechanics

```bash
python3 contrib/compute_submission.py my_pattern.json \
    --output my_record.json \
    --verbose
```

Produces a JSON record with the same schema as
`atlas.json:grounds[i]`. Includes both spring and beam mechanics on
parameter grids. Optionally render thumbnails:

```bash
python3 contrib/compute_submission.py my_pattern.json \
    --output my_record.json \
    --thumbnails-out thumbnails/user/my_pattern \
    --verbose
```

### Step 3: Rank against atlas

```bash
python3 contrib/rank_submission.py my_record.json --atlas atlas.json
```

Produces a human-readable report covering:
- Percentile of `nu_min` and `nu_max` in the catalog distribution
- Five nearest neighbors by Frobenius distance on the C tensor
- Pareto status (is your submission strictly better than any existing
  entry on auxetic-vs-compactness?)
- Duplicate check (does your topology match an existing entry up to
  isomorphism?)

Add `--json` to get machine-readable output.

### Step 4: Add to your local atlas

```bash
python3 contrib/add_to_local_atlas.py my_record.json --atlas atlas.json
```

The visualizer (which loads `atlas.json`) will then show your
submission alongside catalog entries. By default this refuses to add
isomorphic duplicates; pass `--allow-duplicates` to override.

A `.bak` copy of your atlas is created automatically before mutation.

## Community contribution via pull request

1. Fork the repository.
2. Add your submission as `submissions/{family}__{name}.json` (use
   `__` to encode the family/name separator in a flat directory).
3. Open a pull request against the upstream main branch.
4. The CI (`.github/workflows/validate_submissions.yml`) automatically
   runs validation, mechanics, and ranking on every changed submission
   in your PR, and posts the report as a comment.
5. Maintainers review, request changes if needed, and merge. Merged
   submissions are added to the master atlas in the next atlas-rebuild
   cycle.

## Tips for finding good candidates

If you're looking for novel auxetic candidates worth submitting:

- Most promising: graphs with **3- or 6-fold rotational symmetry**,
  which Fowler-Guest 2013 guarantees admit at least one symmetry-detectable
  equiauxetic mechanism. Look for unit cells containing equilateral
  triangle motifs.
- Also good: **chiral structures** (no mirror symmetry). The atlas
  currently has fewer chiral entries than the symmetry analysis
  suggests should exist.
- Mid-cell-size sweet spot: **4×4 unit cells**. Smaller is usually
  rigid; larger is often a stretched version of a smaller pattern.
- Validate against the Pareto front before submitting: if 5 existing
  entries strictly dominate your submission on `(nu_min, cell_area)`,
  it's probably not worth contributing.

## Attribution and licensing

User submissions are accepted under CC-BY 4.0 by default. By submitting
you assert that:
- You have the right to license the topology under CC-BY 4.0.
- The submission is original or derivative of an explicitly cited prior
  source.

Atlas data referencing the TesseLace catalog retains the original Irvine
attribution (CC-BY 4.0).
