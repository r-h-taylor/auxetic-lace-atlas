"""
Adds a full filterable catalog grid + click-to-enlarge modal to
visualizer/index.html. Idempotent: safe to run multiple times.

Usage: from the repo root,
    python3 /path/to/visualizer_catalog_patch.py
"""
import os, re, sys

path = "visualizer/index.html"
if not os.path.isfile(path):
    print("ERROR: visualizer/index.html not found. Run from the repo root.")
    sys.exit(1)

with open(path) as f:
    s = f.read()

# === 1. Modal HTML — append before </body> if not already there ===
modal_html = '''
<!-- Catalog detail modal -->
<div id="ground-modal" class="modal-overlay" hidden>
  <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="modal-title">
    <button class="modal-close" type="button" aria-label="Close">&times;</button>
    <h2 id="modal-title"></h2>
    <div class="modal-body">
      <div class="modal-images">
        <figure>
          <figcaption>pair diagram + thread sketch</figcaption>
          <img id="modal-lace" alt="lace pair diagram" loading="lazy">
        </figure>
        <figure>
          <figcaption>deformed under tension</figcaption>
          <img id="modal-deformed" alt="deformed lace swatch" loading="lazy">
        </figure>
      </div>
      <div id="modal-stats" class="modal-stats"></div>
    </div>
  </div>
</div>
'''

if 'id="ground-modal"' not in s:
    s = s.replace("</body>", modal_html + "\n</body>")

# === 2. Replace the ground-list section's interior so it has search + full grid ===
# Find the existing <section id="ground-list"> block and replace its contents.
ground_list_pat = re.compile(
    r'(<section id="ground-list">)(.*?)(</section>)',
    re.DOTALL,
)
new_ground_list_inner = '''
    <h2>Catalog</h2>
    <div class="catalog-controls">
      <input type="search" id="catalog-search"
             placeholder="filter by family or name (e.g. 3_6 or 2x4_86)…"
             autocomplete="off">
      <select id="catalog-sort">
        <option value="nu_min_beam">sort: ν_min (beam, ascending)</option>
        <option value="nu_min_spring">sort: ν_min (spring k_ang=0.01)</option>
        <option value="family">sort: family / name</option>
        <option value="cell">sort: unit cell size</option>
      </select>
      <label class="catalog-toggle">
        <input type="checkbox" id="catalog-only-auxetic">
        only auxetic (ν<sub>min</sub> &lt; 0)
      </label>
      <span id="catalog-count" class="muted"></span>
    </div>
    <div id="ground-grid"></div>
'''
m = ground_list_pat.search(s)
if m:
    s = s[:m.start()] + m.group(1) + new_ground_list_inner + m.group(3) + s[m.end():]

# === 3. Replace the script with the new behavior ===
# Old script starts at <script> and ends at </script> (single block in this file).
script_pat = re.compile(r'<script>.*?</script>', re.DOTALL)
new_script = '''<script>
// Globals so handlers can reach atlas data without re-fetching
let ATLAS = null;
let SUMMARY = [];

function fmtNu(x) {
  if (x === null || x === undefined) return '-';
  return (x >= 0 ? '+' : '') + x.toFixed(3);
}

function thumbPaths(s) {
  const fam = String(s.family).replace(/[^a-zA-Z0-9_-]/g, '_');
  return {
    lace:     `thumbnails/${fam}/${s.name}/lace.png`,
    deformed: `thumbnails/${fam}/${s.name}/deformed.png`,
  };
}

function renderHeaderStats() {
  const meta = ATLAS.metadata || {};
  const counts = {};
  for (const s of SUMMARY) {
    const cls = s.spring_default_classification || 'unknown';
    counts[cls] = (counts[cls] || 0) + 1;
  }

  document.querySelector('#summary h2').textContent =
    `${meta.n_grounds || 0} grounds in the atlas`;

  let html = '';
  html += '<dl class="stats">';
  html += '<dt>Build date</dt><dd>' + (meta.build_date || 'unknown').slice(0, 10) + '</dd>';
  html += '<dt>Spring k<sub>ang</sub> grid</dt><dd><code>' +
    JSON.stringify(meta.spring_k_ang_grid || []) + '</code></dd>';
  html += '<dt>Beam aspect-ratio grid</dt><dd><code>' +
    JSON.stringify(meta.beam_AR_grid || []) + '</code></dd>';
  html += '</dl>';

  html += '<h3>Classification at default spring parameters</h3>';
  html += '<ul>';
  for (const [cls, n] of Object.entries(counts).sort((a, b) => b[1] - a[1])) {
    const pct = ((n / SUMMARY.length) * 100).toFixed(1);
    html += `<li><strong>${cls}</strong>: ${n} (${pct}%)</li>`;
  }
  html += '</ul>';

  // Top homogeneously-auxetic table (the headline result)
  const topHom = SUMMARY
    .filter(s => s.beam_default_nu_max !== null && s.beam_default_nu_max < 0)
    .sort((a, b) => (a.beam_default_nu_min || 0) - (b.beam_default_nu_min || 0))
    .slice(0, 10);
  html += '<h3>Top homogeneously-auxetic grounds (beam model, AR=10)</h3>';
  html += '<p class="muted">Homogeneous auxetics: ν<sub>max</sub> &lt; 0 in every direction. The 11-of-321 most rigorously auxetic subset; click any to inspect.</p>';
  html += '<table class="auxetic-table"><thead><tr>' +
    '<th>Family</th><th>Name</th><th>ν<sub>min</sub></th><th>ν<sub>max</sub></th><th>Class</th>' +
    '</tr></thead><tbody>';
  for (const s of topHom) {
    html += `<tr class="clickable-row" data-family="${s.family}" data-name="${s.name}">
      <td><code>${s.family}</code></td>
      <td><code>${s.name}</code></td>
      <td>${fmtNu(s.beam_default_nu_min)}</td>
      <td>${fmtNu(s.beam_default_nu_max)}</td>
      <td>${s.beam_default_classification}</td>
    </tr>`;
  }
  html += '</tbody></table>';

  document.getElementById('atlas-info').innerHTML = html;

  // Wire click on table rows to open modal
  for (const tr of document.querySelectorAll('.clickable-row')) {
    tr.addEventListener('click', () => {
      openModal(tr.dataset.family, tr.dataset.name);
    });
  }
}

function renderCatalogGrid() {
  const search = (document.getElementById('catalog-search').value || '').toLowerCase().trim();
  const sortKey = document.getElementById('catalog-sort').value;
  const onlyAux = document.getElementById('catalog-only-auxetic').checked;

  // Filter
  let filtered = SUMMARY.filter(s => {
    if (onlyAux) {
      if (s.beam_default_nu_min === null || s.beam_default_nu_min >= 0) return false;
    }
    if (!search) return true;
    const haystack = `${s.family}/${s.name}`.toLowerCase();
    return haystack.includes(search);
  });

  // Sort
  switch (sortKey) {
    case 'nu_min_beam':
      filtered.sort((a, b) => (a.beam_default_nu_min ?? 1) - (b.beam_default_nu_min ?? 1));
      break;
    case 'nu_min_spring':
      filtered.sort((a, b) => (a.spring_default_nu_min ?? 1) - (b.spring_default_nu_min ?? 1));
      break;
    case 'family':
      filtered.sort((a, b) => {
        const af = `${a.family}/${a.name}`;
        const bf = `${b.family}/${b.name}`;
        return af.localeCompare(bf);
      });
      break;
    case 'cell':
      filtered.sort((a, b) => {
        const ac = (a.n_rows || 0) * (a.n_cols || 0);
        const bc = (b.n_rows || 0) * (b.n_cols || 0);
        if (ac !== bc) return ac - bc;
        return `${a.family}/${a.name}`.localeCompare(`${b.family}/${b.name}`);
      });
      break;
  }

  // Render
  document.getElementById('catalog-count').textContent =
    `showing ${filtered.length} of ${SUMMARY.length}`;

  const grid = document.getElementById('ground-grid');
  let html = '';
  for (const s of filtered) {
    const tp = thumbPaths(s);
    const cls = (s.beam_default_classification || '').replace(/_/g, ' ');
    html += `<figure class="catalog-item" data-family="${s.family}" data-name="${s.name}">
      <img src="${tp.lace}" alt="${s.family}/${s.name}" loading="lazy">
      <figcaption>
        <strong>${s.family}/${s.name}</strong><br>
        <span class="meta">${s.n_rows}×${s.n_cols} cell · ${cls}</span><br>
        ν<sub>min</sub> = ${fmtNu(s.beam_default_nu_min)},
        ν<sub>max</sub> = ${fmtNu(s.beam_default_nu_max)}
      </figcaption>
    </figure>`;
  }
  grid.innerHTML = html;

  for (const f of grid.querySelectorAll('.catalog-item')) {
    f.addEventListener('click', () => {
      openModal(f.dataset.family, f.dataset.name);
    });
  }
}

function openModal(family, name) {
  const ground = ATLAS.grounds.find(g => g.family === family && g.name === name);
  if (!ground) return;
  const summaryEntry = SUMMARY.find(s => s.family === family && s.name === name) || {};
  const tp = thumbPaths({family, name});

  document.getElementById('modal-title').textContent = `${family}/${name}`;
  document.getElementById('modal-lace').src = tp.lace;
  document.getElementById('modal-deformed').src = tp.deformed;

  // Build the stats panel
  const meta = ATLAS.metadata || {};
  const sIdx = meta.spring_default_idx ?? 2;
  const bIdx = meta.beam_default_idx ?? 1;
  const sk = meta.spring_k_ang_grid?.[sIdx];
  const bAR = meta.beam_AR_grid?.[bIdx];

  let html = '';
  html += '<dl class="stats">';
  html += `<dt>Family / Name</dt><dd>${family} / ${name}</dd>`;
  html += `<dt>Period parallelogram</dt><dd>${ground.n_rows} × ${ground.n_cols}</dd>`;
  html += `<dt>Vertices / Edges</dt><dd>${ground.n_vertices} / ${ground.n_edges}</dd>`;
  html += `<dt>Cell area</dt><dd>${(ground.cell_area ?? 0).toFixed(3)}</dd>`;
  html += '</dl>';

  // Spring sweep table
  html += '<h3>Spring model: ν vs angular regularization k<sub>ang</sub></h3>';
  html += '<table class="auxetic-table"><thead><tr><th>k<sub>ang</sub></th><th>ν<sub>min</sub></th><th>ν<sub>max</sub></th><th>Class</th></tr></thead><tbody>';
  const sg = ground.spring || {};
  for (let i = 0; i < (sg.k_ang || []).length; i++) {
    const isDefault = i === sIdx;
    html += `<tr${isDefault ? ' class="default-row"' : ''}>
      <td>${sg.k_ang[i]}</td>
      <td>${fmtNu(sg.nu_min[i])}</td>
      <td>${fmtNu(sg.nu_max[i])}</td>
      <td>${sg.classification[i]}</td>
    </tr>`;
  }
  html += '</tbody></table>';

  // Beam sweep table
  html += '<h3>Beam model: ν vs rod aspect ratio</h3>';
  html += '<table class="auxetic-table"><thead><tr><th>aspect ratio</th><th>ν<sub>min</sub></th><th>ν<sub>max</sub></th><th>Class</th></tr></thead><tbody>';
  const bg = ground.beam || {};
  for (let i = 0; i < (bg.AR || []).length; i++) {
    const isDefault = i === bIdx;
    html += `<tr${isDefault ? ' class="default-row"' : ''}>
      <td>${bg.AR[i]}</td>
      <td>${fmtNu(bg.nu_min[i])}</td>
      <td>${fmtNu(bg.nu_max[i])}</td>
      <td>${bg.classification[i]}</td>
    </tr>`;
  }
  html += '</tbody></table>';

  html += `<p class="muted">Defaults shown highlighted: spring k<sub>ang</sub> = ${sk}, beam AR = ${bAR}.</p>`;

  document.getElementById('modal-stats').innerHTML = html;

  const modal = document.getElementById('ground-modal');
  modal.hidden = false;
  document.body.classList.add('modal-open');
}

function closeModal() {
  document.getElementById('ground-modal').hidden = true;
  document.body.classList.remove('modal-open');
}

async function init() {
  try {
    const resp = await fetch('atlas.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    ATLAS = await resp.json();
    SUMMARY = ATLAS.summary || [];

    renderHeaderStats();
    renderCatalogGrid();

    // Wire up the catalog controls
    document.getElementById('catalog-search').addEventListener('input', renderCatalogGrid);
    document.getElementById('catalog-sort').addEventListener('change', renderCatalogGrid);
    document.getElementById('catalog-only-auxetic').addEventListener('change', renderCatalogGrid);

    // Modal close handlers
    document.querySelector('.modal-close').addEventListener('click', closeModal);
    document.getElementById('ground-modal').addEventListener('click', (e) => {
      if (e.target.id === 'ground-modal') closeModal();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeModal();
    });
  } catch (err) {
    document.querySelector('#summary h2').textContent = 'Atlas not yet built';
    document.getElementById('atlas-info').innerHTML =
      `<p class="error">Could not load atlas.json: <code>${err.message}</code>.<br>
       This page will populate once the deploy workflow has run.</p>`;
  }
}

init();
</script>'''

if not script_pat.search(s):
    print("ERROR: <script> block not found"); sys.exit(1)
s = script_pat.sub(new_script, s)

with open(path, "w") as f:
    f.write(s)
print("patched visualizer/index.html")

# === 4. Append CSS for the new components if not already there ===
css_path = "visualizer/style.css"
with open(css_path) as f:
    css = f.read()

new_css = '''
/* === Catalog controls === */
.catalog-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
  margin: 0.5rem 0 1rem;
  padding: 0.75rem;
  background: rgba(255, 255, 255, 0.5);
  border: 1px solid #d9cfb6;
  border-radius: 4px;
}

.catalog-controls input[type="search"] {
  flex: 1 1 280px;
  min-width: 0;
  padding: 0.4rem 0.7rem;
  border: 1px solid #d9cfb6;
  border-radius: 3px;
  font: inherit;
  background: #fffaf0;
}

.catalog-controls select {
  padding: 0.35rem 0.5rem;
  border: 1px solid #d9cfb6;
  border-radius: 3px;
  font: inherit;
  background: #fffaf0;
}

.catalog-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.92rem;
  color: #4a423a;
}

#catalog-count {
  margin-left: auto;
  font-size: 0.88rem;
}

/* === Catalog item (grid figure) === */
.catalog-item {
  cursor: pointer;
  transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.catalog-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(80, 60, 30, 0.18);
}
.catalog-item .meta {
  font-size: 0.78rem;
  color: #7a7268;
}

.clickable-row {
  cursor: pointer;
}
.clickable-row:hover td {
  background: rgba(255, 255, 255, 0.6);
}

/* === Modal === */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(28, 28, 28, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem 1rem;
  z-index: 100;
  overflow-y: auto;
}

.modal-overlay[hidden] { display: none; }

body.modal-open { overflow: hidden; }

.modal-card {
  background: #f5f0e3;
  border-radius: 6px;
  max-width: 1100px;
  width: 100%;
  max-height: calc(100vh - 4rem);
  overflow-y: auto;
  padding: 1.5rem 1.75rem 2rem;
  position: relative;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
}

.modal-card h2 {
  margin: 0 0 1rem;
  padding-right: 2rem;  /* leave room for close button */
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 1.3rem;
}

.modal-close {
  position: absolute;
  top: 0.75rem;
  right: 1rem;
  background: transparent;
  border: 0;
  font-size: 2rem;
  line-height: 1;
  color: #4a423a;
  cursor: pointer;
  padding: 0.25rem 0.6rem;
  border-radius: 3px;
}
.modal-close:hover { background: rgba(255, 255, 255, 0.5); }

.modal-images {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
  margin-bottom: 1.5rem;
}
.modal-images figure {
  margin: 0;
  background: #fffaf0;
  border: 1px solid #d9cfb6;
  border-radius: 4px;
  padding: 0.5rem;
}
.modal-images img {
  width: 100%;
  display: block;
}
.modal-images figcaption {
  margin-bottom: 0.5rem;
  font-size: 0.85rem;
  color: #5a5249;
  text-align: center;
}

.modal-stats { color: #2a2520; }

.default-row td {
  background: rgba(180, 140, 70, 0.18);
  font-weight: 600;
}

@media (max-width: 700px) {
  .modal-images { grid-template-columns: 1fr; }
}
'''

if "/* === Catalog controls === */" not in css:
    css += "\n" + new_css
    with open(css_path, "w") as f:
        f.write(css)
    print("appended catalog+modal styles to visualizer/style.css")
else:
    print("CSS already contains catalog/modal styles, skipping")
