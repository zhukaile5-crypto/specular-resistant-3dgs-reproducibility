"""Lazy loader for the exact-order V6 CUDA uniform-filter kernel."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from threading import Lock

import numpy as np

_LIBRARY = None
_LOAD_LOCK = Lock()


def _nvcc_path() -> Path:
    candidates = [
        Path(sys.prefix) / "bin" / "nvcc",
        Path(os.environ.get("CUDA_HOME", "")) / "bin" / "nvcc",
    ]
    discovered = shutil.which("nvcc")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("nvcc was not found; activate the nerfstudio environment")


def _load_library():
    import torch

    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    with _LOAD_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        source = Path(__file__).with_name("csrc") / "uniform_filter_cuda.cu"
        source_bytes = source.read_bytes()
        capability = torch.cuda.get_device_capability()
        architecture = f"{capability[0]}{capability[1]}"
        cache_key = hashlib.sha256(
            source_bytes + architecture.encode("ascii")
        ).hexdigest()[:16]
        cache_dir = (
            Path(tempfile.gettempdir()) / "nvidia_project_cuda_filters"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        library_path = cache_dir / f"uniform_filter_{cache_key}.so"
        lock_path = cache_dir / f"uniform_filter_{cache_key}.lock"
        with lock_path.open("w") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            if not library_path.is_file():
                nvcc = _nvcc_path()
                partial = library_path.with_name(
                    f".{library_path.name}.{os.getpid()}.partial"
                )
                command = [
                    str(nvcc),
                    "-shared",
                    "-O3",
                    "--fmad=false",
                    (
                        f"-gencode=arch=compute_{architecture},"
                        f"code=sm_{architecture}"
                    ),
                    "-Xcompiler",
                    "-fPIC",
                    "-o",
                    str(partial),
                    str(source),
                ]
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        "failed to compile exact CUDA uniform filter:\n"
                        + completed.stderr.strip()
                    )
                os.replace(partial, library_path)
        library = ctypes.CDLL(str(library_path))
        function = library.launch_uniform_filter_axis
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        function.restype = ctypes.c_int
        _LIBRARY = library
        return library


def uniform_filter_cuda_exact(
    values: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """Filter the final two axes with SciPy-compatible operation order.

    Any leading dimensions are treated as independent 2-D planes and moved
    through CUDA in one batch.  Each plane still uses the same sequential
    sliding-window accumulation as the single-plane implementation.
    """
    import torch

    if values.ndim < 2:
        raise ValueError("exact CUDA uniform filter requires at least two axes")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    library = _load_library()
    function = library.launch_uniform_filter_axis
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    original_shape = contiguous.shape
    height, width = original_shape[-2:]
    batch = int(np.prod(original_shape[:-2], dtype=np.int64)) or 1
    current = torch.from_numpy(contiguous.reshape(batch, height, width)).to(
        device="cuda", non_blocking=False
    )
    stream = torch.cuda.current_stream()
    for axis in (0, 1):
        output = torch.empty_like(current)
        error = function(
            ctypes.c_void_p(current.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_int64(batch),
            ctypes.c_int64(current.shape[1]),
            ctypes.c_int64(current.shape[2]),
            ctypes.c_int(axis),
            ctypes.c_int(kernel_size),
            ctypes.c_uint64(stream.cuda_stream),
        )
        if error != 0:
            raise RuntimeError(
                f"exact CUDA uniform-filter launch failed with CUDA error {error}"
            )
        current = output
    return current.cpu().numpy().reshape(original_shape).astype(
        np.float32, copy=False
    )
