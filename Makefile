PYTHON ?= python
CONFIG ?= configs/demo.yaml

.PHONY: install demo preprocess gate1 gate2 gate3 phase4 figures manuscript run-all test clean

install:
	pip install -e .

demo:
	$(PYTHON) -m wue_pipeline.cli make-demo --config $(CONFIG)

preprocess:
	$(PYTHON) -m wue_pipeline.cli preprocess --config $(CONFIG)

gate1:
	$(PYTHON) -m wue_pipeline.cli gate1 --config $(CONFIG)

gate2:
	$(PYTHON) -m wue_pipeline.cli gate2 --config $(CONFIG)

gate3:
	$(PYTHON) -m wue_pipeline.cli gate3 --config $(CONFIG)

phase4:
	$(PYTHON) -m wue_pipeline.cli phase4 --config $(CONFIG)

figures:
	$(PYTHON) -m wue_pipeline.cli figures --config $(CONFIG)

manuscript:
	$(PYTHON) -m wue_pipeline.cli manuscript --config $(CONFIG)

run-all: preprocess gate1 gate2 gate3 phase4 figures manuscript

test:
	pytest -q

clean:
	rm -rf data/demo/* data/interim/* data/processed/* results/tables/* results/figures/* results/memos/* results/manuscript/* logs/*
