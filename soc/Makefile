PYTHON ?= .venv/bin/python
CONFIG ?= config/sim_default.json
BUILD_DIR ?= build
CASE_PATH = $(if $(wildcard $(CASE)),$(abspath $(CASE)),$(abspath cases/$(CASE)))

.PHONY: env-check check generate rtl test sim sim-run frozen-vta-smoke vcr-smoke gemm-smoke case-validate case-run run case-firmware case-firmware-run zcu104

env-check:
	$(PYTHON) tools/check_env.py

check:
	$(PYTHON) tools/soc_config.py check --config $(CONFIG)

generate:
	$(PYTHON) tools/soc_config.py generate --config $(CONFIG) --output $(BUILD_DIR)/generated

# Explicit release preparation; never a dependency of sim/case targets.
rtl:
	$(PYTHON) tools/freeze_vta_rtl.py --config $(CONFIG) $(RTL_ARGS)

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

sim: check generate
	@if [ -n "$(CASE)" ] && ls "$(CASE_PATH)"/*_test.c >/dev/null 2>&1; then \
		$(MAKE) case-firmware CASE_DIR="$(CASE_PATH)"; \
	else \
		./scripts/sim_build.sh $(BUILD_DIR)/sim $(CONFIG); \
	fi

sim-run: sim
	./scripts/sim_case_firmware_macos.sh $(BUILD_DIR)/sim/gateware

frozen-vta-smoke:
	./scripts/frozen_vta_smoke.sh

vcr-smoke:
	$(PYTHON) tools/vcr_smoke.py --gateware $(BUILD_DIR)/sim/gateware

gemm-smoke:
	$(PYTHON) tools/gemm_smoke.py --gateware $(BUILD_DIR)/sim/gateware

case-validate:
	$(PYTHON) tools/vta_case.py validate $(CASE)

case-run: case-validate
	$(PYTHON) tools/case_run.py $(CASE) --gateware $(BUILD_DIR)/sim/gateware

run:
	@if [ -z "$(CASE)" ]; then \
		echo "usage: make run CASE=packed-case-or-c-firmware-case" >&2; exit 2; \
	elif [ -f "$(CASE_PATH)/manifest.json" ]; then \
		$(MAKE) case-run CASE="$(CASE_PATH)"; \
	elif ls "$(CASE_PATH)"/*_test.c >/dev/null 2>&1; then \
		$(MAKE) case-firmware-run CASE_DIR="$(CASE_PATH)"; \
	else \
		echo "unsupported case (need manifest.json or *_test.c): $(CASE_PATH)" >&2; exit 2; \
	fi

case-firmware:
	./scripts/sim_build.sh $(BUILD_DIR)/case-$(notdir $(CASE_DIR)) $(CONFIG) $(CASE_DIR)

case-firmware-run:
	./scripts/sim_case_firmware_macos.sh $(BUILD_DIR)/case-$(notdir $(CASE_DIR))/gateware

zcu104: check generate
	$(PYTHON) litex/vta_soc.py --config $(CONFIG) --build-dir $(BUILD_DIR)/zcu104 --target zcu104 --build
