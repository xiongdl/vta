# Single-operator VTA SoC case; usable directly from this directory.
CASE_SOURCE_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
SOC_DIR := $(abspath $(CASE_SOURCE_DIR)/../..)
ifeq ($(abspath $(CURDIR)),$(CASE_SOURCE_DIR))
.PHONY: rtl sim run
rtl:
	$(MAKE) -C $(SOC_DIR) rtl CONFIG=config/sim_default.json
sim:
	$(MAKE) -C $(SOC_DIR) sim CASE=$(CASE_SOURCE_DIR) CONFIG=config/sim_default.json
run:
	$(MAKE) -C $(SOC_DIR) run CASE=$(CASE_SOURCE_DIR) CONFIG=config/sim_default.json
else
SOC_DIRECTORY := $(SOC_DIR)
BIOS_DIRECTORY := $(SOC_DIRECTORY)/.venv/lib/python3.10/site-packages/litex/soc/software/bios
include $(BIOS_DIRECTORY)/Makefile
LSCRIPT = ../../../../../../../../linker/vta_case.ld
CFLAGS += -I$(BIOS_DIRECTORY)
override VPATH := $(CASE_SOURCE_DIR):$(BIOS_DIRECTORY):$(BIOS_DIRECTORY)/cmds:$(CPU_DIRECTORY)
main.o: $(CASE_SOURCE_DIR)/dense_gemm_bias_clip_test.c
	$(compile)
endif
