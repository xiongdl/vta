PYTHON ?= python3
CONFIG ?= config/sim_default.json
BUILD_DIR ?= build

.PHONY: check generate test sim sim-run vcr-smoke gemm-smoke zcu104

check:
	$(PYTHON) tools/soc_config.py check --config $(CONFIG)

generate:
	$(PYTHON) tools/soc_config.py generate --config $(CONFIG) --output $(BUILD_DIR)/generated

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

sim: check generate
	$(PYTHON) litex/vta_soc.py --config $(CONFIG) --build-dir $(BUILD_DIR)/sim --sim

sim-run:
	./scripts/sim_run_macos.sh

vcr-smoke:
	$(PYTHON) tools/vcr_smoke.py --gateware $(BUILD_DIR)/sim/gateware

gemm-smoke:
	$(PYTHON) tools/gemm_smoke.py --gateware $(BUILD_DIR)/sim/gateware

zcu104: check generate
	$(PYTHON) litex/vta_soc.py --config $(CONFIG) --build-dir $(BUILD_DIR)/zcu104 --target zcu104 --build
