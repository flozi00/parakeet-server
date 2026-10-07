"""Regression coverage for NeMo's CUDA gradient-clamp compilation."""
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLAMP = """def clamp_gradient(g, clamp):
    if clamp > 0.0:
        g = min(g, clamp)
        g = max(g, -clamp)
    return g
"""


def patcher():
    script = ROOT / "scripts" / "nemo_cuda_compat.py"
    if not script.is_file():
        pytest.fail("Image lacks the NeMo CUDA gradient-clamp compatibility fix")
    return runpy.run_path(str(script))["patch_kernel"]


def kernel_source():
    return "\n".join(CLAMP.replace("clamp_gradient", name) for name in (
        "compute_grad_kernel", "compute_multiblank_grad_kernel", "compute_tdt_grad_kernel",
    ))


def test_gradient_clamp_passes_real_cuda_argument_validation(tmp_path, monkeypatch):
    import numba
    from numba.core import compiler
    from numba.core.typed_passes import type_inference_stage
    from numba.core.untyped_passes import FixupArgs
    from numba.cuda.descriptor import cuda_target
    from numba.cuda.dispatcher import CUDADispatcher

    def validate_device_arguments(dispatcher, args, return_type=None):
        # Exercise the real compiler pass that fails in production, without a GPU.
        FixupArgs().run_pass({
            "func_ir": compiler.run_frontend(dispatcher.py_func), "args": args,
            "flags": SimpleNamespace(force_pyobject=False),
        })
        raise AssertionError("Unexpected device overload after the clamp fix")

    monkeypatch.setattr(CUDADispatcher, "compile_device", validate_device_arguments)
    context = cuda_target.typing_context
    context.refresh()

    def infer(function):
        return type_inference_stage(
            context, cuda_target.target_context, compiler.run_frontend(function),
            (numba.types.float32, numba.types.float64), None,
        )

    original = {}
    exec(CLAMP, original)
    with pytest.raises(TypeError, match="Signature mismatch: 2 argument types given"):
        infer(original["clamp_gradient"])

    path = tmp_path / "gpu_rnnt_kernel.py"
    path.write_text(kernel_source())
    patcher()(path)
    fixed = {}
    exec(path.read_text(), fixed)
    for name, function in fixed.items():
        if name.startswith("compute_"):
            assert infer(function).return_type == numba.types.float64


@pytest.mark.parametrize("g,clamp", [
    (-2.0, 1.0), (-1.0, 1.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0),
    (float("inf"), 1.0), (-float("inf"), 1.0), (float("nan"), 1.0),
    (-0.0, 1.0), (2.0, 0.0), (2.0, -1.0), (2.0, float("nan")),
])
def test_clamp_preserves_original_numerical_behavior(tmp_path, g, clamp):
    original = {}
    exec(CLAMP, original)
    path = tmp_path / "gpu_rnnt_kernel.py"
    path.write_text(kernel_source())
    patcher()(path)
    fixed = {}
    exec(path.read_text(), fixed)
    expected = original["clamp_gradient"](g, clamp)
    for name, function in fixed.items():
        if name.startswith("compute_"):
            result = function(g, clamp)
            if math.isnan(expected):
                assert math.isnan(result)
            else:
                assert result == expected
                assert math.copysign(1.0, result) == math.copysign(1.0, expected)


def test_unexpected_upstream_source_is_not_modified(tmp_path):
    path = tmp_path / "gpu_rnnt_kernel.py"
    path.write_text(CLAMP)
    with pytest.raises(RuntimeError, match="Expected three"):
        patcher()(path)
    assert path.read_text() == CLAMP
