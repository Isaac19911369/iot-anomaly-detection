.PHONY: install data benchmark train serve test lint docker

install:
	pip install -e ".[dev,serve,tracking]"

data:
	iot-anomaly download

benchmark: data
	iot-anomaly -v benchmark

train: data
	iot-anomaly train --model conv_ae

serve:
	iot-anomaly serve

test:
	pytest

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

docker:
	docker compose run --rm trainer && docker compose up -d api
