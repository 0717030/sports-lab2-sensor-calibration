# src/viz.py
from typing import Callable, Dict
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

_ACTIONS: Dict[str, Callable] = {}

def register(action_name: str):
    def deco(fn: Callable):
        _ACTIONS[action_name] = fn
        return fn
    return deco

def _savefig(outdir: Path, name: str):
    p = outdir / f"{name}.png"
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    return p

# ===========================
# ✅ 作業需要的最小繪圖集合
# ===========================
def save_required_plots(outdir, df, a_world, v_world, x_world, gyro_dps, gyro_angle_deg):
    import matplotlib.pyplot as plt
    t = df["t_sec"].to_numpy()

    def plot_xyz(t, arr, title, ylabel, filename):
        plt.figure(figsize=(12,6))
        # 依你的需求：x=紅, y=綠, z=藍
        plt.plot(t, arr[:,0], label="x", color="red")
        plt.plot(t, arr[:,1], label="y", color="green")
        plt.plot(t, arr[:,2], label="z", color="blue")
        plt.title(title)
        plt.xlabel("t (sec)")
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.25)
        _savefig(outdir, filename)

    # === Accelerometer (world, corrected) ===
    plot_xyz(t, a_world, "Accelerometer (world, corrected)", "a (m/s²)", "acc_after_world")

    # === Velocity ===
    plot_xyz(t, v_world, "Velocity (world, integrated)", "v (m/s)", "vel_after_world")

    # === Displacement ===
    plot_xyz(t, x_world, "Displacement (world, integrated)", "x (m)", "pos_after_world")

    # === Gyro ω-T ===
    plot_xyz(t, gyro_dps, "Gyroscope Angular Velocity", "ω (deg/s)", "gyro_VT_deg_per_s")

    # === Gyro θ-T ===
    plot_xyz(t, gyro_angle_deg, "Gyroscope Angle (integrated)", "θ (deg)", "gyro_XT_deg")


# ===========================
# 針對特定動作的補充圖（保持原本設計）
# ===========================
def save_action_plots(outdir, action, df, fs, ori, a_world, v_world, x_world):
    fn = _ACTIONS.get(action)
    if fn:
        fn(outdir, df, fs, ori, a_world, v_world, x_world)

@register("pendulum")
def _pendulum(outdir, df, fs, ori, a_world, v_world, x_world):
    t = df["t_sec"].to_numpy()
    plt.figure(figsize=(10,4))
    plt.plot(t, np.rad2deg(ori["euler"][:,2]))
    plt.title("Pendulum Yaw [deg]"); plt.xlabel("Time [s]")
    _savefig(outdir, "pendulum_yaw")

@register("elevator")
def _elevator(outdir, df, fs, ori, a_world, v_world, x_world):
    t = df["t_sec"].to_numpy()
    plt.figure(figsize=(10,4)); plt.plot(t, a_world[:,2]); plt.title("Elevator A_z [m/s^2]"); plt.xlabel("Time [s]")
    _savefig(outdir, "elevator_Az")
    plt.figure(figsize=(10,4)); plt.plot(t, v_world[:,2]); plt.title("Elevator V_z [m/s]"); plt.xlabel("Time [s]")
    _savefig(outdir, "elevator_Vz")
    plt.figure(figsize=(10,4)); plt.plot(t, x_world[:,2]); plt.title("Elevator Z [m]"); plt.xlabel("Time [s]")
    _savefig(outdir, "elevator_Z")

@register("square")
def _square(outdir, df, fs, ori, a_world, v_world, x_world):
    plt.figure(figsize=(5,5))
    plt.plot(x_world[:,0], x_world[:,1])
    plt.axis('equal'); plt.title("Planar Path (X vs Y)")
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    _savefig(outdir, "square_xy")

def save_before_plots(outdir, df):
    """
    Draw raw (pre-calibration) XYZ in one figure each (x=red, y=green, z=blue),
    and integrate naively to get v/x. Also plot raw gyro ω/θ.
    """
    import matplotlib.pyplot as plt
    from src.core import integrate_naive, gyro_tracks_raw
    t = df["t_sec"].to_numpy()

    # --- raw accelerometer in body frame ---
    a_raw = df[["acc_x","acc_y","acc_z"]].to_numpy()

    def plot_xyz(t, arr, title, ylabel, filename):
        plt.figure(figsize=(12,6))
        plt.plot(t, arr[:,0], label="x", color="red")
        plt.plot(t, arr[:,1], label="y", color="green")
        plt.plot(t, arr[:,2], label="z", color="blue")
        plt.title(title)
        plt.xlabel("t (sec)")
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.25)
        _savefig(outdir, filename)

    # A-T (raw, body frame)
    plot_xyz(t, a_raw, "Accelerometer (raw, body frame)", "a (m/s²)", "acc_before_body")

    # V-T & X-T from raw (naive integration)
    v_raw, x_raw = integrate_naive(t, a_raw)
    plot_xyz(t, v_raw, "Velocity (raw, naive integration, body frame)", "v (m/s)", "vel_before_body")
    plot_xyz(t, x_raw, "Displacement (raw, naive integration, body frame)", "x (m)", "pos_before_body")

    # Gyro (raw)
    gyro_dps_raw, gyro_ang_deg_raw = gyro_tracks_raw(df)
    plot_xyz(t, gyro_dps_raw, "Gyroscope Angular Velocity (raw)", "ω (deg/s)", "gyro_VT_before_deg_per_s")
    plot_xyz(t, gyro_ang_deg_raw, "Gyroscope Angle (raw, integrated)", "θ (deg)", "gyro_XT_before_deg")
