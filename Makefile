IMAGE ?= branchgrpo:sglang-verl
CONFIG ?= configs/grpo_sglang.yaml

.PHONY: install test data-smoke dry-run docker-build docker-run

install:
	pip install -r requirements.txt -e .

test:
	pytest -q

data-smoke:
	python scripts/prepare_math_data.py --local-json data/smoketest_20.json --output-dir data/smoketest --train-limit 18 --val-limit 2

dry-run:
	python scripts/launch_verl_grpo.py --config $(CONFIG) --dry-run

docker-build:
	IMAGE=$(IMAGE) ./scripts/docker_build.sh

docker-run:
	IMAGE=$(IMAGE) ./scripts/docker_run.sh bash

