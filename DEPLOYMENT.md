# Deployment Guide

This guide walks through getting the Auxetic Lace Atlas deployed to
GitHub Pages, including the custom domain step.

## Prerequisites

- A GitHub account (and `git` installed locally)
- A domain you've purchased or are about to purchase (~$12/yr)
- The repo skeleton from this drop (see `auxetic_lace_repo/`)

---

## Step 1: Initialize the repository

From the directory containing the unpacked skeleton:

```bash
cd auxetic_lace_repo
git init
git add .
git commit -m "Initial commit: atlas pipeline + visualizer skeleton"
```

Verify the structure looks right:

```bash
git ls-files | head -20
# Should include README.md, pyproject.toml, src/auxetic_lace/*.py, etc.
```

## Step 2: Create the GitHub repo

On GitHub:

1. Go to https://github.com/new
2. **Owner**: your account (or an organization you own)
3. **Repository name**: `auxetic-lace-atlas` (or whatever you prefer; the
   workflows will figure out the right URL automatically)
4. **Visibility**: Public (required for free GitHub Pages)
5. **Do NOT** initialize with README, .gitignore, or license — we already
   have those locally
6. Click "Create repository"

GitHub will show you the push commands. Use them:

```bash
git remote add origin https://github.com/YOUR_USERNAME/auxetic-lace-atlas.git
git branch -M main
git push -u origin main
```

Update the README's GitHub URLs:

```bash
# Replace {your-username} with your actual GitHub username
sed -i.bak "s|{your-username}|YOUR_USERNAME|g" README.md visualizer/index.html pyproject.toml
rm *.bak visualizer/*.bak
git commit -am "Set GitHub username in URLs"
git push
```

## Step 3: Run the pipeline locally first

Before relying on the CI to build everything, verify the full pipeline
works on your machine:

```bash
pip install -e .
make atlas  # or the individual steps:
            # auxetic-lace-scrape && auxetic-lace-thumbnails && auxetic-lace-build-atlas
```

Expected output:
- `tesselace_catalog/` populated with 321 .txt files + manifest.csv (~1 min)
- `tesselace_catalog/thumbnails/` populated with 642 PNGs (~5-10 min)
- `docs/atlas.json` ~10-15 MB (~5-10 min)
- `docs/thumbnails/` mirrors `tesselace_catalog/thumbnails/`

Sanity-check by serving locally:
```bash
make serve
# Open http://localhost:8000
```

The placeholder visualizer should show:
- Total grounds count (should be 321)
- Classification breakdown (~167 directionally auxetic, ~11 homogeneously)
- Top 10 most-auxetic grounds table
- Thumbnail grid for top 6

If anything looks off, fix it locally before going to CI.

## Step 4: Enable GitHub Pages

In your repo on GitHub:

1. **Settings → Pages**
2. **Source**: GitHub Actions
3. Save (this is automatic — no button to click after selecting Source)

## Step 5: Trigger the first deploy manually

The Pages workflow runs on schedule (Sunday 06:00 UTC) and on certain
push events, but you'll want to trigger it manually for the first run:

1. Go to **Actions** tab
2. Click **Build atlas and deploy to Pages** in the left sidebar
3. Click **Run workflow** dropdown → **Run workflow** button
4. Wait ~10-15 minutes for the build to complete (catalog scraping,
   thumbnail rendering, mechanics computation, atlas build, deploy)

If the build succeeds, the **environments** section in your repo
sidebar will show **github-pages** with a link to the live site:
`https://YOUR_USERNAME.github.io/auxetic-lace-atlas/`

Visit it and confirm the placeholder visualizer loads. The atlas
summary panel should populate after a moment.

## Step 6: Set up the custom domain

Once you've purchased your domain (e.g. `auxetic-lace.org`):

### 6a. Configure DNS at your registrar

Add these DNS records at your domain registrar's control panel:

For an apex domain (`auxetic-lace.org`), add four `A` records:
```
@   A   185.199.108.153
@   A   185.199.109.153
@   A   185.199.110.153
@   A   185.199.111.153
```

For a `www` subdomain, add a `CNAME` record:
```
www  CNAME  YOUR_USERNAME.github.io
```

(These are GitHub Pages' standard IPs as of 2024; verify at
https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site
in case they've changed.)

### 6b. Add the CNAME file to your repo

```bash
echo "auxetic-lace.org" > docs/CNAME  # use your actual domain
git add docs/CNAME
git commit -m "Add custom domain CNAME"
git push
```

The deploy workflow respects `docs/CNAME` and preserves it in the Pages
artifact. After the next deploy (manual or scheduled), GitHub Pages will
serve the site at your custom domain.

### 6c. Configure HTTPS

In **Settings → Pages**, after DNS propagates (can take up to 24 hours):
- The **Custom domain** field should auto-populate from `docs/CNAME`
- Click **Save**
- **Enforce HTTPS** checkbox should appear once GitHub finishes
  provisioning the certificate (Let's Encrypt; another ~24 hours)
- Check the box

## Step 7: Verify the contribution flow

Test the PR-based contribution workflow with a sample submission:

```bash
git checkout -b test-submission
cp submission_examples/triangular.json submissions/test_triangular.json
git add submissions/
git commit -m "Test submission: triangular"
git push -u origin test-submission
```

Then on GitHub, open a pull request from the `test-submission` branch
to `main`. Within ~2 minutes, the **Validate submissions** workflow
should run and post a comment on the PR with:
- Schema validation
- Mechanics computation summary
- Ranking against the deployed atlas

Once you've verified it works, close the PR (don't merge — it was just a
test). Or merge it and watch your test submission appear in the live
atlas after the next deploy.

## Step 8: Schedule check

The deploy workflow runs every Sunday at 06:00 UTC. To verify that's
configured:

1. **Settings → Actions → General → Workflow permissions**: should be
   "Read and write permissions" (needed for Pages deploy)
2. **Actions tab → Build atlas and deploy to Pages**: in the schedule
   should show "Every Sunday at 06:00 UTC"

To trigger a rebuild between scheduled runs (e.g. after merging a
contribution PR), manually run the workflow from the Actions tab.

---

## Troubleshooting

### "Pages build failed: artifact too large"

GitHub Pages has a 1 GB artifact limit. If `atlas.json` + thumbnails
exceeds that, options:
- Trim the parameter sweeps in `build_atlas.py` (smaller `SPRING_K_ANG_GRID`
  or `BEAM_AR_GRID`)
- Drop the `u_strain` field from the JSON (forces visualizer to
  recompute deformation, but cuts atlas.json size by ~5x)
- Switch to lazy-loading: split atlas.json into `atlas_summary.json` +
  `atlas_grounds/{name}.json` and have the visualizer fetch on demand

### "Custom domain check failed"

DNS hasn't propagated yet. Check with `dig auxetic-lace.org +short`
from your terminal — it should return the four GitHub IPs. Wait 24
hours and retry.

### "scrape_tesselace.py failed in CI"

The scraper hits `d-bl.github.io`. If GitHub Actions blocks that
domain, you can pre-scrape the catalog locally and check it into the
repo (don't gitignore `tesselace_catalog/` in that case). It's only
~2 MB total. The Action will skip the scrape step if the cache is
populated.

### "submissions/*.json fails CI but works locally"

Check that the JSON keys are quoted and there are no trailing commas
(some JSON parsers are stricter than others). Also verify the file is
under `submissions/`, not `submission_examples/`.

---

## Summary checklist

- [ ] Skeleton unpacked locally
- [ ] `pip install -e .` and `pytest -q` pass (27 tests)
- [ ] `make atlas` produces a valid `docs/atlas.json`
- [ ] `make serve` shows the placeholder visualizer with stats
- [ ] GitHub repo created, code pushed
- [ ] GitHub Pages enabled (Source: GitHub Actions)
- [ ] First manual `workflow_dispatch` run completed successfully
- [ ] Default domain `https://USER.github.io/REPO/` loads
- [ ] Domain purchased
- [ ] DNS A records configured at registrar
- [ ] `docs/CNAME` committed
- [ ] HTTPS enforced in Settings → Pages
- [ ] Test PR opened and CI comment received

Once all boxes checked, you have a live atlas at your custom domain
that rebuilds weekly and accepts community contributions via PR. Next
session can focus entirely on building out the real interactive
visualizer in `visualizer/`.
