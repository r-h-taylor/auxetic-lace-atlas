# Auxetic Lace Atlas

An interactive atlas of mechanical metamaterials derived from bobbin
lace topologies. Live site: https://auxetic-lace.github.io/atlas/
([custom domain pending](#deployment))

This project takes the public TesseLace catalog of 321 bobbin lace
ground patterns ([Irvine, 2016](https://dspace.library.uvic.ca/items/867c403c-4f45-4c54-89d1-1c8d138dfe92))
and computes the full 2D mechanical response of each topology under
two interpretations: an axial-spring framework (pin-jointed limit) and
an Euler-Bernoulli beam frame (3D-printable structural realization).
Of the 321 topologies, **52% are directionally auxetic** and **3.4%
are homogeneously auxetic**, with the strongest candidates achieving
ν<sub>min</sub> = -2.49.

## What this repo contains

```
.
├── src/auxetic_lace/              # Python package
│   ├── parse_to_graph.py          # TesseLace .txt -> directed graph parser
│   ├── mechanics.py               # Spring (axial + perpendicular) homogenization
│   ├── mechanics_beam.py          # Euler-Bernoulli frame homogenization
│   ├── build_atlas.py             # Top-level atlas pipeline
│   ├── build_thumbnails.py        # Pre-render PNG views
│   ├── render_lace_*.py           # Pair-diagram and deformed-overlay renderers
│   ├── scrape_tesselace.py        # Catalog scraper
│   ├── canonical/                 # Canonical labeling for graph isomorphism
│   └── contrib/                   # User contribution pipeline
├── visualizer/                    # Static HTML/JS frontend
├── submissions/                   # Community-contributed graphs (PR target)
├── submission_examples/           # Sample submission JSONs
├── tests/                         # Unit tests
├── .github/workflows/             # CI: validate PRs, weekly atlas rebuild
└── docs/                          # GitHub Pages output (built artifacts, not in repo)
```

## Running locally

```bash
git clone https://github.com/{your-username}/auxetic-lace-atlas.git
cd auxetic-lace-atlas
pip install -e .
```

Build the atlas data (one-time, ~15 min):
```bash
auxetic-lace-scrape                      # ~1 min, downloads catalog
auxetic-lace-thumbnails --verbose        # ~5-10 min, renders 642 PNGs
auxetic-lace-build-atlas --verbose       # ~5-10 min, runs mechanics
```

Or in one command:
```bash
make atlas
```

The visualizer is served from `docs/`. You can open it locally:
```bash
cp -r visualizer/* docs/
cd docs && python3 -m http.server 8000
# Open http://localhost:8000
```

## Submitting your own graph

The atlas accepts user-contributed periodic 2D directed graphs.
See [`src/auxetic_lace/contrib/README.md`](src/auxetic_lace/contrib/README.md)
for the full submission workflow, schema, and validation/ranking tools.

Quick version (after `pip install -e .`):
```bash
# 1. Author your graph as JSON (see submission_examples/)
# 2. Validate & compute mechanics
auxetic-lace-validate my_pattern.json
auxetic-lace-compute my_pattern.json --output my_record.json
# 3. Rank against the atlas
auxetic-lace-rank my_record.json --atlas docs/atlas.json
# 4. Either add locally or open a PR
auxetic-lace-add my_record.json --atlas docs/atlas.json
# OR:
git checkout -b add-my-pattern
cp my_pattern.json submissions/
git add submissions/my_pattern.json && git commit -m "Add my_pattern"
git push  # then open PR — CI will validate automatically
```

## Mechanical models

Each ground is computed under two models:

- **Spring** (`mechanics.py`): axial spring at each edge with stiffness
  k = 1/L, plus optional perpendicular regularization k<sub>ang</sub>.
  Setting k<sub>ang</sub>=0 recovers the strict pin-jointed limit; finite
  k<sub>ang</sub> reveals which grounds are robustly auxetic.
- **Beam** (`mechanics_beam.py`): full Euler-Bernoulli frame element
  with axial EA + flexural EI, parameterized by rod aspect ratio.
  Validated to converge to the spring model in the high-aspect-ratio
  limit (Δ < 1e-6 at AR=200 for canonical test cases).

Both are computed on a parameter grid and stored in `atlas.json`. The
visualizer toggles between them with a slider.

## Deployment

The live visualizer is served from GitHub Pages. The `docs/` folder is
built and deployed automatically by
[.github/workflows/deploy_pages.yml](.github/workflows/deploy_pages.yml):

- **Trigger**: weekly schedule (Sunday 06:00 UTC) + manual via
  `workflow_dispatch`. Not on every push, since the build takes 10-15
  minutes.
- **Steps**: scrape the TesseLace catalog (cached), render thumbnails
  (cached), build atlas, copy visualizer files, deploy to Pages.
- **Custom domain**: when a domain is purchased, drop a `CNAME` file in
  `docs/` (committed to the workflow's deployment artifact) and
  configure DNS. Currently default `*.github.io`.

## Contributing

PRs welcome for:
- New graph topologies (add to `submissions/`, see `contrib/README.md`)
- Bug fixes in the mechanics code
- Visualizer improvements
- New mechanical models (e.g., nonlinear, large-deformation)

The CI (`.github/workflows/validate_submissions.yml`) runs validation,
mechanics, and ranking on every submission PR and posts the report as a
comment.

## Citation

If you use this atlas in published work, please cite:

```
@article{taylor2026auxetic,
  title={Auxetic Lace Metamaterials: A High-Throughput Mechanical Screen
         of Bobbin Lace Topologies for 3D-Printed Structural Frames},
  author={Taylor, R. H.},
  year={2026},
  note={Manuscript in preparation}
}
```

The underlying lace ground catalog is courtesy of:

```
@phdthesis{irvine2016thesis,
  title={Lace Tessellations: A mathematical model for bobbin lace and
         an exhaustive combinatorial search for patterns},
  author={Irvine, Veronika},
  school={University of Victoria},
  year={2016},
  url={https://dspace.library.uvic.ca/items/867c403c-4f45-4c54-89d1-1c8d138dfe92}
}
```

## License

- **Code** (everything under `src/`, `contrib/`, `visualizer/`,
  `tests/`): MIT
- **Atlas data** (`atlas.json` and thumbnails, derived from the
  TesseLace catalog): CC-BY 4.0, requires Irvine attribution
- **User submissions** in `submissions/`: CC-BY 4.0 by default; submitters
  may specify a different compatible license in the `provenance` field.

See [LICENSE](LICENSE) for full text.
