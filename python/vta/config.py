"""VTA compiler configuration independent from TVM's in-tree VTA package."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VTAConfig:
    """Hardware properties needed by Relay capability checks and legalization."""

    target: str
    batch: int
    block_in: int
    block_out: int
    input_bits: int
    weight_bits: int
    accumulator_bits: int

    @classmethod
    def from_json(cls, path: Optional[str] = None):
        if path is None:
            path = Path(__file__).resolve().parents[2] / "config" / "vta_config.json"
        with open(path, encoding="utf-8") as config_file:
            raw = json.load(config_file)
        log_block_in = raw.get("LOG_BLOCK_IN", raw["LOG_BLOCK"])
        log_block_out = raw.get("LOG_BLOCK_OUT", raw["LOG_BLOCK"])
        return cls(
            target=raw["TARGET"],
            batch=1 << raw["LOG_BATCH"],
            block_in=1 << log_block_in,
            block_out=1 << log_block_out,
            input_bits=1 << raw["LOG_INP_WIDTH"],
            weight_bits=1 << raw["LOG_WGT_WIDTH"],
            accumulator_bits=1 << raw["LOG_ACC_WIDTH"],
        )

