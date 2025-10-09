# src/pipeline.py (pendulum-enhanced version)
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import src.core as core
import src.io_utils as io_mod
import src.viz as viz

@dataclass
class PipelineConfig:
    csv: str
    out: str = "results"
    action: str = "custom"           # 'pendulum' | 'elevator' | 'square' | 'custom'
    alpha: float = 0.05              # slightly stronger accel correction
    gyro_thresh_dps: float = 5.0
    min_stationary_sec: float = 0.5
    use_mag: bool = False
    export_calibrated_csv: bool = False


def _estimate_biases_strong(df, fs):
    """Use only first 3s, auto-find most stable 1s window."""
    import pandas as pd
    gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
    t = df["t_sec"].to_numpy()
    mask3s = t <= 3.0
    if not mask3s.any():
        raise RuntimeError("no samples within first 3 seconds")
    gyro3s = gyro[mask3s]
    t3s = t[mask3s]
    N = len(t3s)
    win_len = int(fs * 1.0)  # 1 second window
    if win_len < 3 or win_len >= N:
        win_len = max(3, N // 3)
    mags = np.sqrt((np.rad2deg(gyro3s) ** 2).sum(axis=1))
    mean_mag = np.convolve(mags, np.ones(win_len)/win_len, mode="valid")
    best_i = int(np.argmin(mean_mag))
    sel = slice(best_i, best_i + win_len)
    stat = np.zeros(len(df), bool)
    stat[np.where(mask3s)[0][sel]] = True

    a_mean = df.loc[stat, ["acc_x","acc_y","acc_z"]].mean().to_numpy()
    g = 9.80665
    if np.linalg.norm(a_mean) > 1e-6:
        g_vec = g * (a_mean / np.linalg.norm(a_mean))
    else:
        g_vec = np.array([0.0, 0.0, g])
    acc_bias = a_mean - g_vec

    gyro_bias = df.loc[stat, ["gyro_x","gyro_y","gyro_z"]].mean().to_numpy()
    mag_bias = (
        df.loc[stat, ["mag_x","mag_y","mag_z"]].mean().to_numpy()
        if {"mag_x","mag_y","mag_z"}.issubset(df.columns)
        else np.zeros(3)
    )
    return {"acc": acc_bias, "gyro": gyro_bias, "mag": mag_bias}, stat


def run(cfg: PipelineConfig) -> Dict[str, Any]:
    df, fs = io_mod.load_csv(cfg.csv)
    outdir = io_mod.ensure_outdir(cfg.out)

    # ===== Specialized pendulum bias estimation =====
    if cfg.action == "pendulum":
        biases, stationary_mask = _estimate_biases_strong(df, fs)
    else:
        biases, stationary_mask = core.estimate_biases(
            df, fs,
            gyro_thresh_dps=cfg.gyro_thresh_dps,
            min_stationary_sec=cfg.min_stationary_sec
        )

    ori = core.estimate_orientation(df, fs, biases, alpha=cfg.alpha, use_mag=cfg.use_mag)
    a_world = core.to_world_linear_acc(df, ori, biases)
    v_world, x_world = core.integrate_velocity_position(df["t_sec"].to_numpy(), a_world, stationary_mask)
    gyro_dps, gyro_angle_deg = core.gyro_tracks(df, biases)

    if cfg.export_calibrated_csv:
        io_mod.export_timeseries(outdir,
            t=df["t_sec"].to_numpy(),
            a_world=a_world, v_world=v_world, x_world=x_world,
            gyro_dps=gyro_dps, gyro_angle_deg=gyro_angle_deg
        )

    viz.save_before_plots(outdir, df)
    viz.save_required_plots(outdir, df, a_world, v_world, x_world, gyro_dps, gyro_angle_deg)
    viz.save_action_plots(outdir, cfg.action, df, fs, ori, a_world, v_world, x_world)

    io_mod.write_manifest(outdir, cfg, biases, fs)
    return {"biases": biases, "fs": fs, "ori": ori}
