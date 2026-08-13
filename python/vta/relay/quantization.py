"""QNN compatibility normalization kept entirely outside Apache TVM."""

import numpy as np
import tvm
from tvm import relay


def fixed_point_ratio(input_scale, output_scale):
    """Return an exact power-of-two shift for a scalar scale ratio.

    Positive values mean left shift; negative values mean right shift.  Other
    ratios need TVM-compatible rounding/multiplier semantics and are rejected.
    """
    ratio = float(input_scale) / float(output_scale)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Quantization scales must be finite and positive")
    shift = int(round(np.log2(ratio)))
    if not np.isclose(ratio, 2.0 ** shift):
        raise ValueError("VTA exact fixed-point normalization requires a power-of-two ratio")
    return shift


class _UniformScaleNormalizer(relay.ExprMutator):
    def visit_call(self, call):
        rewritten = super().visit_call(call)
        if not isinstance(rewritten.op, tvm.ir.Op) or rewritten.op.name != "qnn.requantize":
            return rewritten
        args = list(rewritten.args)
        for index in (1, 3):
            if not isinstance(args[index], relay.Constant):
                continue
            values = np.asarray(args[index].data.numpy())
            if values.size <= 1:
                continue
            if not np.allclose(values, values.reshape(-1)[0]):
                raise ValueError(
                    "TVM 0.17 cannot type non-uniform per-channel qnn.requantize scales"
                )
            args[index] = relay.const(values.reshape(-1)[0], str(values.dtype))
        return relay.Call(rewritten.op, args, rewritten.attrs, rewritten.type_args, rewritten.span)


def normalize_qnn_scales(mod):
    """Collapse uniform per-channel scales before TVM 0.17 InferType.

    Non-uniform vectors are rejected explicitly; silently averaging them would
    change quantized model semantics.
    """
    updated = tvm.IRModule(mod.functions, mod.type_definitions, mod.attrs)
    for global_var, function in list(updated.functions.items()):
        if isinstance(function, relay.Function):
            updated[global_var] = _UniformScaleNormalizer().visit(function)
    return updated
