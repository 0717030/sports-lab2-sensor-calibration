# src/pipeline.py
from dataclasses import dataclass
from typing import Dict, Any
import src.core as core
import src.io_utils as io_mod  # local io.py (not stdlib)
import src.viz as viz

@dataclass
class PipelineConfig:
    csv: str
    out: str = "results"
    action: str = "custom"           # 'pendulum' | 'elevator' | 'square' | 'custom'
    alpha: float = 0.02              # complementary filter accel blend (0~1)
    gyro_thresh_dps: float = 3.0     # stationary detection threshold [deg/s]
    min_stationary_sec: float = 0.5  # min duration to treat as stationary
    use_mag: bool = False            # yaw correction by magnetometer
    export_calibrated_csv: bool = False

def run(cfg: PipelineConfig) -> Dict[str, Any]:
    df, fs = io_mod.load_csv(cfg.csv)
    outdir = io_mod.ensure_outdir(cfg.out)

    biases, stationary_mask = core.estimate_biases(
        df, fs,
        gyro_thresh_dps=cfg.gyro_thresh_dps,
        min_stationary_sec=cfg.min_stationary_sec
    )

    # Orientation via complementary filter (gyro used to correct accel)
    ori = core.estimate_orientation(
        df, fs, biases, alpha=cfg.alpha, use_mag=cfg.use_mag
    )

    # Corrected (world-frame linear acceleration) + integrations
    a_world = core.to_world_linear_acc(df, ori, biases)
    v_world, x_world = core.integrate_velocity_position(df["t_sec"].to_numpy(), a_world, stationary_mask)

    # Gyroscope tracks
    gyro_dps, gyro_angle_deg = core.gyro_tracks(df, biases)

    if cfg.export_calibrated_csv:
        io_mod.export_timeseries(outdir,
            t=df["t_sec"].to_numpy(),
            a_world=a_world, v_world=v_world, x_world=x_world,
            gyro_dps=gyro_dps, gyro_angle_deg=gyro_angle_deg
        )

    # ✅ 只輸出作業需要的圖
    viz.save_required_plots(outdir, df, a_world, v_world, x_world, gyro_dps, gyro_angle_deg)
    viz.save_action_plots(outdir, cfg.action, df, fs, ori, a_world, v_world, x_world)

    io_mod.write_manifest(outdir, cfg, biases, fs)
    return {"biases": biases, "fs": fs, "ori": ori}
