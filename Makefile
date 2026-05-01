# Auxetic Lace Atlas — common tasks

.PHONY: help install atlas thumbnails scrape demo test clean serve

help:
	@echo "Common tasks:"
	@echo "  make install      Install the package in editable mode"
	@echo "  make atlas        Full pipeline: scrape + thumbnails + atlas"
	@echo "  make scrape       Download the TesseLace catalog"
	@echo "  make thumbnails   Render thumbnails for all grounds"
	@echo "  make demo         Build a small demo atlas (no catalog needed)"
	@echo "  make test         Run unit tests"
	@echo "  make serve        Serve the visualizer locally on port 8000"
	@echo "  make clean        Remove built artifacts (catalog, atlas, thumbnails)"

install:
	pip install -e .[dev]

scrape:
	auxetic-lace-scrape --output tesselace_catalog/

thumbnails:
	auxetic-lace-thumbnails --catalog tesselace_catalog/ --verbose

atlas: scrape thumbnails
	mkdir -p docs
	auxetic-lace-build-atlas \
	    --catalog tesselace_catalog/ \
	    --output docs/atlas.json \
	    --thumbnail-dir thumbnails \
	    --verbose
	mkdir -p docs/thumbnails
	cp -r tesselace_catalog/thumbnails/* docs/thumbnails/
	cp -r visualizer/* docs/
	@echo ""
	@echo "Atlas built at docs/atlas.json ($$( stat -c%s docs/atlas.json 2>/dev/null || stat -f%z docs/atlas.json ) bytes)"
	@echo "Thumbnails at docs/thumbnails/"
	@echo "Run 'make serve' to view locally at http://localhost:8000"

demo:
	python3 -m auxetic_lace.build_demo_atlas

test:
	pytest -v

serve:
	@if [ ! -f docs/atlas.json ]; then \
	    echo "atlas.json not found. Run 'make atlas' first or 'make demo' for a smoke-test atlas."; \
	    exit 1; \
	fi
	cd docs && python3 -m http.server 8000

clean:
	rm -rf tesselace_catalog/
	rm -f docs/atlas.json
	rm -rf docs/thumbnails/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	@echo "Clean complete."
