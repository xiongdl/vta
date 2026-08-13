"""Logical tensor adapters and a small mixed CPU/VTA execution plan."""

import ctypes

import numpy as np
import tvm
from tvm import relay, rpc
from tvm.contrib import utils

from ..libinfo import find_libvta


def _load_local_module(artifact):
    temp = utils.tempdir()
    path = temp.relpath("kernel.o")
    artifact.module.save(path)
    remote = rpc.LocalSession()
    remote.upload(path)
    return remote, remote.load_module("kernel.o")


def run_artifact(artifact, data):
    """Run a dense or conv2d artifact using logical NumPy tensor layouts."""
    from .conv2d import Conv2DArtifact
    from .dense import DenseArtifact

    env = __import__("vta").get_env()
    remote, module = _load_local_module(artifact)
    device = remote.ext_dev(0)
    data = np.asarray(data, dtype=env.inp_dtype)
    if isinstance(artifact, DenseArtifact):
        batch, features = artifact.input_shape
        packed_data = data.reshape(
            batch // env.BATCH, env.BATCH, features // env.BLOCK_IN, env.BLOCK_IN
        ).transpose(0, 2, 1, 3)
        packed_output_shape = (
            batch // env.BATCH,
            artifact.output_shape[1] // env.BLOCK_OUT,
            env.BATCH,
            env.BLOCK_OUT,
        )
        logical = lambda value: value.reshape(artifact.output_shape)
    elif isinstance(artifact, Conv2DArtifact):
        batch, channels, height, width = artifact.input_shape
        physical_channels = artifact.packed_input_shape[1] * env.BLOCK_IN
        padded = np.zeros((batch, physical_channels, height, width), dtype=env.inp_dtype)
        padded[:, :channels] = data
        packed_data = padded.reshape(
            batch // env.BATCH,
            env.BATCH,
            physical_channels // env.BLOCK_IN,
            env.BLOCK_IN,
            height,
            width,
        ).transpose(0, 2, 4, 5, 1, 3)
        packed_output_shape = artifact.packed_output_shape

        def logical(value):
            physical = value.transpose(0, 4, 1, 5, 2, 3).reshape(
                batch,
                packed_output_shape[1] * env.BLOCK_OUT,
                artifact.output_shape[2],
                artifact.output_shape[3],
            )
            return physical[:, : artifact.output_shape[1]]

    else:
        raise TypeError(f"Unsupported VTA artifact type: {type(artifact).__name__}")
    output = tvm.nd.empty(packed_output_shape, env.out_dtype, device)
    try:
        module(
            tvm.nd.array(packed_data, device),
            tvm.nd.array(artifact.packed_weight, device),
            tvm.nd.array(artifact.packed_bias, device),
            output,
        )
        return logical(output.numpy())
    finally:
        # Uop caches contain module-local signature pointers. A mixed plan can
        # unload one generated module before invoking the next, so reset the
        # per-thread VTA command queue at each logical region boundary.
        runtime = ctypes.CDLL(find_libvta("libvta_fsim")[0])
        runtime.VTARuntimeShutdown()


class ExecutionPlan:
    """Executable partitioned Relay graph with explicit logical VTA boundaries."""

    def __init__(self, module, artifacts):
        self.module = module
        self.artifacts = artifacts

    def run(self, **inputs):
        main = self.module["main"]
        bindings = {param: np.asarray(inputs[param.name_hint]) for param in main.params}
        global_names = {var: var.name_hint for var in self.module.get_global_vars()}

        def evaluate(expr, local):
            if isinstance(expr, relay.Var):
                return local[expr]
            if isinstance(expr, relay.Constant):
                return expr.data.numpy()
            if isinstance(expr, relay.Let):
                value = evaluate(expr.value, local)
                return evaluate(expr.body, {**local, expr.var: value})
            if isinstance(expr, relay.Tuple):
                return tuple(evaluate(field, local) for field in expr.fields)
            if isinstance(expr, relay.TupleGetItem):
                return evaluate(expr.tuple_value, local)[expr.index]
            if not isinstance(expr, relay.Call):
                raise TypeError(f"Unsupported Relay expression in VTA execution plan: {type(expr)}")
            args = [evaluate(arg, local) for arg in expr.args]
            if isinstance(expr.op, relay.GlobalVar):
                return run_artifact(self.artifacts[global_names[expr.op]], args[0])
            if isinstance(expr.op, relay.Function):
                return evaluate(expr.op.body, dict(zip(expr.op.params, args)))
            name = expr.op.name
            if name == "add":
                return args[0] + args[1]
            if name == "subtract":
                return args[0] - args[1]
            if name == "multiply":
                return args[0] * args[1]
            if name == "clip":
                return np.clip(args[0], expr.attrs.a_min, expr.attrs.a_max)
            if name == "reshape":
                return np.reshape(args[0], tuple(int(x) for x in expr.attrs.newshape))
            if name == "nn.relu":
                return np.maximum(args[0], 0)
            raise ValueError(f"Unsupported CPU operator between VTA regions: {name}")

        return evaluate(main.body, bindings)
