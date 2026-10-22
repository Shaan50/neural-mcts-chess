"""Export a training checkpoint for deployment.

Keeps only the (EMA) weights and model config, optionally in fp16 (halves
the file size; weights are cast back to fp32 on load), and can additionally
write TorchScript or ONNX graphs for runtimes without the Python model code.

    python -m azchess.tools.export runs/sl_medium/best.pt weights/azchess.pt --half
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from azchess.encoding.board import NUM_PLANES
from azchess.model.inference import load_checkpoint


def export(src: str, dst: str, half: bool = False, torchscript: bool = False, onnx: bool = False) -> None:
    model, ckpt = load_checkpoint(src, "cpu")
    state = {k: (v.half() if half and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    out = {
        "model": state,
        "ema": None,
        "model_config": ckpt["model_config"],
        "step": ckpt.get("step"),
        "source": str(src),
    }
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print(f"[export] {dst} ({Path(dst).stat().st_size / 1e6:.1f} MB, {model.num_parameters() / 1e6:.2f}M params)")
    example = torch.zeros(1, NUM_PLANES, 8, 8)
    if torchscript:
        ts = torch.jit.trace(model, example)
        p = Path(dst).with_suffix(".ts")
        ts.save(str(p))
        print(f"[export] {p}")
    if onnx:
        p = Path(dst).with_suffix(".onnx")
        torch.onnx.export(
            model,
            example,
            str(p),
            input_names=["planes"],
            output_names=["policy", "value"],
            dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
            opset_version=17,
        )
        print(f"[export] {p}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--half", action="store_true")
    p.add_argument("--torchscript", action="store_true")
    p.add_argument("--onnx", action="store_true")
    a = p.parse_args(argv)
    export(a.src, a.dst, a.half, a.torchscript, a.onnx)


if __name__ == "__main__":
    main()
