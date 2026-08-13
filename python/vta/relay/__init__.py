"""Relay passes for automatic VTA region discovery and partitioning."""

from .partition import partition_for_vta
from .patterns import pattern_table

__all__ = ["partition_for_vta", "pattern_table"]

