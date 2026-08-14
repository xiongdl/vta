#!/usr/bin/env python3
"""Export torchvision's official quantized ResNet-18 state dict to portable NPZ.

The exporter deliberately does not instantiate the quantized model, because
torchvision's quantized execution backend is unavailable on macOS.  Quantized
tensors in the official state dict can still be decoded without executing an
operator.  The resulting archive contains plain NumPy arrays only and can be
consumed by the TVM/VTA environment without importing PyTorch.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


URL = "https://download.pytorch.org/models/quantized/resnet18_fbgemm_16fa66dd.pth"
SHA256_PREFIX = "16fa66dd"


def _array(value):
    if not isinstance(value, torch.Tensor):
        return None
    if value.is_quantized:
        return value.int_repr().cpu().numpy()
    return value.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    state = torch.hub.load_state_dict_from_url(URL, map_location="cpu", progress=True)
    arrays = {}
    quantization = {}
    for name, value in state.items():
        array = _array(value)
        if array is None:
            continue
        arrays[name] = array
        if value.is_quantized:
            scheme = str(value.qscheme())
            entry = {"scheme": scheme}
            if value.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
                entry.update(
                    axis=int(value.q_per_channel_axis()),
                    scales=value.q_per_channel_scales().cpu().numpy().tolist(),
                    zero_points=value.q_per_channel_zero_points().cpu().numpy().tolist(),
                )
            else:
                entry.update(scale=float(value.q_scale()), zero_point=int(value.q_zero_point()))
            quantization[name] = entry
    metadata = {
        "source": URL,
        "checkpoint_sha256_prefix": SHA256_PREFIX,
        "tensor_count": len(arrays),
        "quantization": quantization,
    }
    arrays["__metadata_json__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({
        "output": str(args.output),
        "sha256": digest,
        "source": URL,
        "checkpoint_sha256_prefix": SHA256_PREFIX,
        "tensor_count": len(arrays) - 1,
        "quantized_tensor_count": len(quantization),
    }, indent=2))


if __name__ == "__main__":
    main()
