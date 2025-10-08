# src/io_utils.py
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

REQUIRED = ["t_sec","acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]

def load_csv(path: str) -> Tuple[pd.DataFrame, float]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    df = pd.read_csv(p)
    for col in REQUIRED:
        if col not in df.columns:
            raise ValueError(f"CSV missing column: {col}")
    t = df["t_sec"].to_numpy()
    dt = np.diff(t)
    fs = 1.0 / np.median(dt[dt > 0])
    return df, fs

def ensure_outdir(out: str) -> Path:
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    return p

def export_timeseries(outdir: Path, **named_arrays):
    for name, arr in named_arrays.items():
        import pandas as pd
        import numpy as np
        df = pd.DataFrame(np.asarray(arr))
        df.to_csv(outdir / f"{name}.csv", index=False)

def write_manifest(outdir: Path, cfg, biases, fs: float):
    info = {
        "fs": fs,
        "biases": {k: list(map(float, v)) for k, v in biases.items()},
        "config": vars(cfg)
    }
    (outdir / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
