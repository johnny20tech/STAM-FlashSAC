import os
import warnings
from pathlib import Path


def configure_runtime() -> tuple[Path, Path]:
    package_dir = Path(__file__).resolve().parent
    repo_dir = package_dir.parent
    runtime_dir = repo_dir / "runtime"
    tmp_dir = runtime_dir / "tmp"
    cache_dir = runtime_dir / "cache"
    mpl_dir = runtime_dir / "matplotlib"
    inductor_cache_dir = runtime_dir / "torchinductor_compile_no_cudagraphs"
    for path in (tmp_dir, cache_dir, mpl_dir, inductor_cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.update({
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "JAX_DEFAULT_MATMUL_PRECISION": "float32",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        "TMPDIR": str(tmp_dir),
        "TEMP": str(tmp_dir),
        "TMP": str(tmp_dir),
        "XDG_CACHE_HOME": str(cache_dir),
        "MPLCONFIGDIR": str(mpl_dir),
        "TORCHINDUCTOR_CACHE_DIR": str(inductor_cache_dir),
        "TORCHINDUCTOR_USE_CUDAGRAPHS": "0",
    })
    warnings.filterwarnings("ignore")
    return repo_dir, runtime_dir


def configure_torch_runtime(torch) -> None:
    import torch._inductor.config as inductor_config

    inductor_config.triton.cudagraphs = False
    inductor_config.triton.cudagraph_trees = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
