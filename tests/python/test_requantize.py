"""Bit-exact TFLite/CMSIS-NN requantization parameter and rounding tests."""

import math

import pytest

import vta


def _cmsis_reference(value, multiplier, shift):
    if shift > 0:
        value = ((value & 0xFFFFFFFF) << shift) & 0xFFFFFFFF
        value = value - (1 << 32) if value & 0x80000000 else value
    high = (value * multiplier + (1 << 30)) >> 31
    exponent = max(-shift, 0)
    if not exponent:
        return high
    mask = (1 << exponent) - 1
    divided = high >> exponent
    threshold = (mask >> 1) + int(divided < 0)
    return divided + int((high & mask) > threshold)


@pytest.mark.parametrize("scale", [2.0 ** -31, 0.00390625, 0.2, 0.625, 1.0, 1.5, 3.75])
def test_tflite_quantize_multiplier(scale):
    multiplier, shift = vta.quantize_multiplier(scale)
    assert 0 <= multiplier <= (1 << 31) - 1
    assert -31 <= shift <= 30
    assert math.isclose(math.ldexp(multiplier, shift - 31), scale, rel_tol=1e-9)


@pytest.mark.parametrize("scale", [0.2, 0.625, 1.0, 1.5, 3.75])
@pytest.mark.parametrize("value", [-(1 << 30), -65537, -17, -5, -1, 0, 1, 5, 17, 65537, (1 << 30) - 1])
def test_cmsis_nn_requantize_default_rounding(value, scale):
    multiplier, shift = vta.quantize_multiplier(scale)
    assert vta.cmsis_nn_requantize(value, multiplier, shift) == _cmsis_reference(
        value, multiplier, shift
    )
