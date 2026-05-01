"""
TesseLace catalog scraper.

Downloads the TesseLace bobbin lace ground catalog from
    https://d-bl.github.io/tesselace-to-gf/

These ground patterns are the result of research by Veronika Irvine
(see arXiv:1406.1532 and her PhD thesis), enumerated using a
combinatorial search over 2-regular doubly-periodic toroidal directed
graphs. They are released under CC-BY 4.0; ATTRIBUTION REQUIRED in any
downstream use:

    Veronika Irvine, "TesseLace: Algorithmically designed lace
    tessellations", https://d-bl.github.io/tesselace-to-gf/
    Licensed CC-BY 4.0.

This scraper is a polite citizen:
- Identifies itself in the User-Agent
- Rate-limits to ~2 requests per second
- Caches downloaded files locally so re-runs are no-ops
- Writes a manifest CSV cataloging every ground

Usage:
    python3 scrape_tesselace.py                # default: download all
    python3 scrape_tesselace.py --output cat   # custom output directory
    python3 scrape_tesselace.py --dry-run      # show what WOULD be downloaded
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Set, Tuple


BASE_URL = "https://d-bl.github.io/tesselace-to-gf"

# Pages on the catalog site that link to .txt template files
INDEX_PAGES = [
    "index.html",          # traditional grounds (cloth, diamond, kat, rose, bias)
    "5.html",              # pentagon-hole grounds
    "6.html",              # hexagon-hole grounds
    "7.html",              # heptagon-hole grounds
    "8_9_10.html",         # octagon, nonagon, decagon
    "fouche_3x4.html",     # Burden of Excess community sampler
]

# Politeness
USER_AGENT = (
    "AuxeticLaceScraper/0.1 "
    "(research; contact via github.com/d-bl/tesselace-to-gf)"
)
REQUEST_DELAY_S = 0.5     # ~2 requests / second
MAX_RETRIES = 3
RETRY_DELAY_S = 2.0


def _http_get(url: str) -> str:
    """Fetch a URL with retries and polite headers. Returns response body."""
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_S * (attempt + 1))
            continue
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} attempts: "
                       f"{last_exc}")


# Pattern: matches paths like "tl/3_5/2x4_111.txt" or "/tesselace-to-gf/tl/.../*.txt"
TXT_LINK_RE = re.compile(
    r"(?:https?://[^/]*)?(?:/tesselace-to-gf)?/?(tl/[A-Za-z0-9_]+/[A-Za-z0-9_]+\.txt)"
)


def extract_txt_paths(html: str) -> List[str]:
    """Find all .txt template references in an index page's HTML.
    Returns paths relative to BASE_URL (e.g., 'tl/3_5/2x4_111.txt')."""
    matches = TXT_LINK_RE.findall(html)
    # Deduplicate while preserving order of first appearance
    seen = set()
    out = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def discover_all_txt_paths(verbose: bool = True) -> List[str]:
    """Crawl all index pages and collect the union of .txt template paths."""
    all_paths: List[str] = []
    seen: Set[str] = set()
    for page in INDEX_PAGES:
        url = f"{BASE_URL}/{page}"
        if verbose:
            print(f"  crawling {url}", flush=True)
        try:
            html = _http_get(url)
        except RuntimeError as e:
            print(f"    WARNING: could not fetch {url}: {e}", flush=True)
            continue
        paths = extract_txt_paths(html)
        new = [p for p in paths if p not in seen]
        for p in new:
            seen.add(p)
            all_paths.append(p)
        if verbose:
            print(f"    {len(paths)} txt links ({len(new)} new)", flush=True)
        time.sleep(REQUEST_DELAY_S)
    return all_paths


def parse_template(text: str) -> Optional[Dict]:
    """Parse a TesseLace .txt file into structured form.

    File format:
        <KEYWORD>\\t<n_rows>\\t<n_cols>
        [x1,y1,x2,y2,x3,y3]\\t[...]\\t...
        [x1,y1,...]\\t...

    Returns a dict:
        {
            'keyword':   str ('Lattice Path' or 'CHECKER'),
            'n_rows':    int,
            'n_cols':    int,
            'arcs':      list of (x1,y1,x2,y2,x3,y3) tuples (3 lattice points
                         each, the arc passing through them in order)
        }
    Returns None if the file fails to parse.
    """
    text = text.strip()
    if not text:
        return None

    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return None

    # First line: header. The keyword can be "Lattice Path" (with space) or
    # "CHECKER", followed by tab-separated n_rows, n_cols.
    header = lines[0]
    parts = header.split("\t")
    if len(parts) < 3:
        return None
    keyword = parts[0].strip()
    try:
        n_rows = int(parts[1])
        n_cols = int(parts[2])
    except ValueError:
        return None

    # Remaining lines: arcs. Each arc is in form [x1,y1,x2,y2,x3,y3]
    # separated by tabs.
    arc_re = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,"
                        r"\s*(-?\d+)\s*,\s*(-?\d+)\s*,"
                        r"\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
    arcs: List[Tuple[int, int, int, int, int, int]] = []
    for line in lines[1:]:
        for m in arc_re.finditer(line):
            arcs.append(tuple(int(x) for x in m.groups()))

    if not arcs:
        return None

    return {
        "keyword": keyword,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "arcs": arcs,
    }


def download_template(rel_path: str, output_dir: str,
                      overwrite: bool = False) -> Optional[str]:
    """Download a single .txt file. Returns local path on success, None on
    failure. Caches: skips download if file already exists and overwrite=False.
    """
    local_path = os.path.join(output_dir, rel_path)
    if os.path.exists(local_path) and not overwrite:
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = f"{BASE_URL}/{rel_path}"
    try:
        text = _http_get(url)
    except RuntimeError as e:
        print(f"    WARNING: could not download {url}: {e}", flush=True)
        return None
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(text)
    return local_path


def family_from_path(rel_path: str) -> str:
    """Extract the family directory name from a path like 'tl/3_5/2x4_111.txt'."""
    parts = rel_path.split("/")
    if len(parts) >= 3 and parts[0] == "tl":
        return parts[1]
    return ""


def name_from_path(rel_path: str) -> str:
    """Extract the ground name (without extension) from rel_path."""
    base = os.path.basename(rel_path)
    return os.path.splitext(base)[0]


def build_manifest(rel_paths: List[str], output_dir: str) -> List[Dict]:
    """Read every downloaded template, parse, and produce a manifest."""
    manifest: List[Dict] = []
    for rel in rel_paths:
        local = os.path.join(output_dir, rel)
        if not os.path.exists(local):
            continue
        try:
            with open(local, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        parsed = parse_template(text)
        if parsed is None:
            print(f"    WARNING: failed to parse {rel}", flush=True)
            continue
        manifest.append({
            "name": name_from_path(rel),
            "family": family_from_path(rel),
            "rel_path": rel,
            "local_path": local,
            "source_url": f"{BASE_URL}/{rel}",
            "keyword": parsed["keyword"],
            "n_rows": parsed["n_rows"],
            "n_cols": parsed["n_cols"],
            "n_arcs": len(parsed["arcs"]),
        })
    return manifest


def write_manifest_csv(manifest: List[Dict], path: str) -> None:
    if not manifest:
        return
    fieldnames = list(manifest[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in manifest:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(
        description="Scrape the TesseLace bobbin lace ground catalog.")
    ap.add_argument("--output", "-o", default="tesselace_catalog",
                    help="output directory (default: tesselace_catalog)")
    ap.add_argument("--dry-run", action="store_true",
                    help="discover paths but don't download")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-download files that already exist locally")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on number of files (for testing)")
    args = ap.parse_args()

    out = args.output
    os.makedirs(out, exist_ok=True)

    print("=" * 70)
    print("TesseLace catalog scraper")
    print("=" * 70)
    print(f"  source:      {BASE_URL}")
    print(f"  output:      {out}")
    print(f"  index pages: {len(INDEX_PAGES)}")
    print()

    # 1) Crawl index pages
    print("Step 1: discovering .txt template files...", flush=True)
    rel_paths = discover_all_txt_paths()
    print(f"  found {len(rel_paths)} unique template files", flush=True)

    if args.limit is not None:
        rel_paths = rel_paths[: args.limit]
        print(f"  capped to {len(rel_paths)} (--limit)", flush=True)

    if args.dry_run:
        print("\nDry run — would download:")
        for p in rel_paths:
            print(f"  {p}")
        return

    # 2) Download files
    print("\nStep 2: downloading template files...", flush=True)
    n_ok = 0
    n_skipped = 0
    n_fail = 0
    for i, rel in enumerate(rel_paths, 1):
        local = os.path.join(out, rel)
        already = os.path.exists(local) and not args.overwrite
        local_after = download_template(rel, out, overwrite=args.overwrite)
        if local_after is None:
            n_fail += 1
            status = "FAIL"
        elif already:
            n_skipped += 1
            status = "cached"
        else:
            n_ok += 1
            status = "ok"
            time.sleep(REQUEST_DELAY_S)
        if i % 25 == 0 or i == len(rel_paths):
            print(f"  {i}/{len(rel_paths)} ({n_ok} ok, {n_skipped} cached, "
                  f"{n_fail} fail)", flush=True)

    # 3) Build manifest
    print("\nStep 3: parsing templates and building manifest...", flush=True)
    manifest = build_manifest(rel_paths, out)
    print(f"  parsed {len(manifest)} templates", flush=True)

    manifest_path = os.path.join(out, "manifest.csv")
    write_manifest_csv(manifest, manifest_path)
    print(f"  manifest written to {manifest_path}", flush=True)

    # Summary
    print("\nSummary:")
    families: Dict[str, int] = {}
    for m in manifest:
        families[m["family"]] = families.get(m["family"], 0) + 1
    for fam in sorted(families):
        print(f"  family '{fam}': {families[fam]} grounds")
    print()
    sizes: Dict[Tuple[int, int], int] = {}
    for m in manifest:
        k = (m["n_rows"], m["n_cols"])
        sizes[k] = sizes.get(k, 0) + 1
    print("  by unit-cell size (n_rows × n_cols):")
    for size in sorted(sizes):
        print(f"    {size[0]}×{size[1]}: {sizes[size]} grounds")

    print("\nDone.")
    print(f"\nAttribution required for downstream use:\n"
          f"  Veronika Irvine, 'TesseLace: Algorithmically designed lace\n"
          f"  tessellations', {BASE_URL}\n"
          f"  Licensed CC-BY 4.0.")


if __name__ == "__main__":
    main()
