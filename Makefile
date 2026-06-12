.PHONY: install test smoke paper docs clean-results clean

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

docs:
	cd docs/latex && pdflatex documentation.tex && bibtex documentation && pdflatex documentation.tex && pdflatex documentation.tex

clean-results:
	$(PYTHON) scripts/clean_results.py

clean: clean-results
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
