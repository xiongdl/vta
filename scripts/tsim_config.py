"""Emit shell-safe TSIM parameters derived from a VTA JSON config."""

import json
import sys


with open(sys.argv[1], encoding="utf-8") as config_file:
    cfg = json.load(config_file)

batch = 1 << cfg["LOG_BATCH"]
block = 1 << cfg["LOG_BLOCK"]
inp_bits = 1 << cfg["LOG_INP_WIDTH"]
wgt_bits = 1 << cfg["LOG_WGT_WIDTH"]
acc_bits = 1 << cfg["LOG_ACC_WIDTH"]
values = {
    "VTA_TSIM_BATCH": batch,
    "VTA_TSIM_BLOCK_IN": block,
    "VTA_TSIM_BLOCK_OUT": block,
    "VTA_TSIM_INP_BITS": inp_bits,
    "VTA_TSIM_WGT_BITS": wgt_bits,
    "VTA_TSIM_ACC_BITS": acc_bits,
    "VTA_TSIM_OUT_BITS": inp_bits,
    "VTA_TSIM_BUS_BITS": 1 << cfg["LOG_BUS_WIDTH"],
    "VTA_TSIM_UOP_DEPTH": (1 << cfg["LOG_UOP_BUFF_SIZE"]) // 4,
    "VTA_TSIM_INP_DEPTH": (1 << cfg["LOG_INP_BUFF_SIZE"]) * 8 // (batch * block * inp_bits),
    "VTA_TSIM_WGT_DEPTH": (1 << cfg["LOG_WGT_BUFF_SIZE"]) * 8 // (block * block * wgt_bits),
    "VTA_TSIM_ACC_DEPTH": (1 << cfg["LOG_ACC_BUFF_SIZE"]) * 8 // (batch * block * acc_bits),
    "VTA_TSIM_OUT_DEPTH": (1 << cfg["LOG_INP_BUFF_SIZE"]) * 8 // (batch * block * inp_bits),
}
for key, value in values.items():
    print(key, value)
