"""Top-level compiler orchestration owned by tvm-vta."""

from ..relay import partition_for_vta
from .dense import compile_partitioned_dense


def compile(mod, params=None, config=None):
    """Partition a Relay module and compile its single supported VTA region."""
    partitioned = partition_for_vta(mod, params=params, config=config)
    return compile_partitioned_dense(partitioned)
