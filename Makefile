PYTHON ?= python
OUTDIR ?= data

SCRIPTS := 01_krein_reduction.py \
           02_theorem_a_branch.py \
           03_theorem_a_convergence.py \
           04_power_balance.py \
           05_parity_gamma.py \
           06_gauge_monodromy.py

CSVS := krein_identities.csv \
        eps_scan.csv \
        theorem_a_branch.csv \
        theorem_a_convergence.csv \
        power_balance.csv \
        parity_gamma.csv \
        gauge_monodromy.csv

.DEFAULT_GOAL := help
.PHONY: help all test test-slow clean

help:
	@echo "nonhermitian-vk -- reproducibility targets"
	@echo
	@echo "  make all        run scripts 01-06 in order, writing CSV + JSON to $(OUTDIR)/"
	@echo "  make test       run the fast regression tests"
	@echo "  make test-slow  run the slow regression tests (scripts 02, 03, 06)"
	@echo "  make clean      remove $(OUTDIR)/*.csv and $(OUTDIR)/*.json"
	@echo "  make help       this message (default target)"
	@echo
	@echo "Variables: PYTHON=$(PYTHON)  OUTDIR=$(OUTDIR)"

all:
	@set -e; \
	mkdir -p $(OUTDIR); \
	for s in $(SCRIPTS); do \
	    echo "=== scripts/$$s"; \
	    $(PYTHON) scripts/$$s --outdir $(OUTDIR); \
	    echo; \
	done; \
	echo "=== summary"; \
	for f in $(CSVS); do \
	    if [ -f $(OUTDIR)/$$f ]; then \
	        printf '  %-30s %s data rows\n' "$(OUTDIR)/$$f" "$$(awk 'END {print NR-1}' $(OUTDIR)/$$f)"; \
	    else \
	        echo "  MISSING: $(OUTDIR)/$$f"; exit 1; \
	    fi; \
	done

test:
	PYTHONPATH=src $(PYTHON) -m pytest

test-slow:
	PYTHONPATH=src $(PYTHON) -m pytest -m slow

clean:
	rm -f $(OUTDIR)/*.csv $(OUTDIR)/*.json
