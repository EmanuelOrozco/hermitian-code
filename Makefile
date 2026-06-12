.PHONY: install test smoke paper publication docs clean-results clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	pytest tests/ -v

smoke:
	$(PYTHON) scripts/run_reproduction.py --profile smoke

paper:
	$(PYTHON) scripts/run_reproduction.py --profile paper

publication:
	$(PYTHON) scripts/clean_results.py
	$(PYTHON) scripts/run_reproduction.py --profile publication
	cd docs/latex && latexmk -pdf -interaction=nonstopmode -halt-on-error documentation.tex
	mkdir -p results/publication
	cp docs/latex/documentation.pdf results/publication/publication.pdf

docs:
	cd docs/latex && latexmk -pdf -interaction=nonstopmode -halt-on-error documentation.tex

clean-results:
	$(PYTHON) scripts/clean_results.py

clean: clean-results
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	rm -f docs/latex/*.aux docs/latex/*.bbl docs/latex/*.blg docs/latex/*.fls
	rm -f docs/latex/*.fdb_latexmk docs/latex/*.log docs/latex/*.out docs/latex/*.toc
	rm -f docs/latex/*.synctex.gz docs/latex/*.pdf
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
