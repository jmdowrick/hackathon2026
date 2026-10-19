VENV := .venv/bin
.PHONY: help test test-all toy ala2 clean
help:
	@echo "make test      - fast unit+regression tests (~3s)"
	@echo "make test-all  - all tests incl. training (~10s)"
	@echo "make toy       - toy system, exact ground truth (~20s)"
	@echo "make ala2      - real alanine dipeptide (~45s)"
test:
	$(VENV)/python -m pytest tests/ -q -m "not slow"
test-all:
	$(VENV)/python -m pytest tests/ -q
toy:
	$(VENV)/python -c "from ala2bg import pipeline; pipeline.run(out_dir='runs/toy', provider='toy', provider_kw=dict(n_steps=60_000, n_traj=8, dt=0.5, D=0.01, barrier=12.0))"
ala2:
	$(VENV)/python -c "from ala2bg import pipeline; pipeline.run(out_dir='runs/ala2', provider='mdshare', provider_kw=dict(cache_dir='data', dt=1.0))"
clean:
	rm -rf runs/* && find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
