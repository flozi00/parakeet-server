"""Build-time workaround for NeMo's variadic CUDA min/max overloads."""
import importlib.util
from pathlib import Path
import re


def patch_kernel(path: Path) -> None:
    source = path.read_text()
    fixed, count = re.subn(
        r"(?m)^([ \t]*)g = min\(g, clamp\)\n\1g = max\(g, -clamp\)$",
        r"\1g = clamp if clamp < g else g\n\1g = -clamp if -clamp > g else g",
        source,
    )
    if count != 3:
        raise RuntimeError(f"Expected three NeMo CUDA gradient clamps; found {count}.")
    path.write_text(fixed)


if __name__ == "__main__":
    spec = importlib.util.find_spec("nemo")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("NeMo must be installed before applying its CUDA compatibility fix.")
    nemo_root = Path(spec.submodule_search_locations[0])
    patch_kernel(nemo_root / "collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt_kernel.py")
