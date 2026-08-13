"""Relay passes for automatic VTA region discovery and partitioning."""

from .partition import partition_for_vta
from .patterns import pattern_table
from .quantization import fixed_point_ratio, normalize_qnn_scales

__all__ = ["fixed_point_ratio", "normalize_qnn_scales", "partition_for_vta", "pattern_table"]
