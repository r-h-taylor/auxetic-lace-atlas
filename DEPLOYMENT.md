# Deployment

This repo's `docs/` folder is a static site served via GitHub Pages.

## Initial setup

1. Push this repository to GitHub (private or public).
2. In **Settings → Pages**, set:
   - **Source**: Deploy from a branch
   - **Branch**: `main` (or whatever your default branch is)
   - **Folder**: `/docs`
3. Wait ~1 minute for the first build.
4. Visit `https://YOUR_USERNAME.github.io/auxetic-lace-atlas/`.

## Updating the atlas

The atlas is rebuilt locally with `make atlas` (see Makefile) and
committed to `docs/atlas.json`. Pushing to the default branch triggers
a Pages rebuild automatically.

## Custom domain (optional)

If you have a domain, add a `CNAME` file:

```bash
echo "your-domain.org" > docs/CNAME
git add docs/CNAME
git commit -m "Add custom domain"
git push
```

Then point your domain's DNS at GitHub Pages following GitHub's
[custom domain docs](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site).

## Repository size note

The `docs/thumbnails/` directory is stored via Git LFS. Cloning the
repo with thumbnails requires `git lfs` installed:

```bash
brew install git-lfs        # or your platform's equivalent
git lfs install             # one-time per machine
git clone https://github.com/USER/auxetic-lace-atlas.git
```

Without git-lfs, the clone will succeed but thumbnails will be
placeholder pointer files instead of actual PNGs.
