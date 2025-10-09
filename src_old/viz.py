# src/viz.py (pendulum-enhanced)
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


# ====================================================
# ✅ General required plots (unchanged)
# ====================================================
def save_required_plots(outdir, df, a_world, v_world, x_world, gyro_dps, gyro_angle_deg):
    import matplotlib.pyplot as plt
    t = df["t_sec"].to_numpy()

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

    plot_xyz(t, a_world, "Accelerometer (world, corrected)", "a (m/s²)", "acc_after_world")
    plot_xyz(t, v_world, "Velocity (world, integrated)", "v (m/s)", "vel_after_world")
    plot_xyz(t, x_world, "Displacement (world, integrated)", "x (m)", "pos_after_world")
    plot_xyz(t, gyro_dps, "Gyroscope Angular Velocity", "ω (deg/s)", "gyro_VT_deg_per_s")
    plot_xyz(t, gyro_angle_deg, "Gyroscope Angle (integrated)", "θ (deg)", "gyro_XT_deg")


# ====================================================
# 🕑 Pendulum-specific improved visualization
# ====================================================
@register("pendulum")
def _pendulum(outdir, df, fs, ori, a_world, v_world, x_world):
    t = df["t_sec"].to_numpy()

    # 自動偵測主要擺動軸
    var3 = a_world.var(axis=0)
    main_axis = np.argmax(var3)
    axis_name = ["x", "y", "z"][main_axis]

    # 畫 pitch 而非 yaw
    roll, pitch, yaw = np.rad2deg(ori["euler"].T)
    plt.figure(figsize=(10,4))
    plt.plot(t, pitch, label="pitch", color="purple")
    plt.title(f"Pendulum Pitch [deg] (main axis ≈ {axis_name.upper()})")
    plt.xlabel("Time [s]")
    plt.ylabel("Pitch (deg)")
    plt.grid(True, alpha=0.3)
    _savefig(outdir, "pendulum_pitch")

    # 額外：在主要軸畫加速度+速度+位移
    plt.figure(figsize=(10,4))
    plt.plot(t, a_world[:,main_axis], label=f"a_{axis_name}")
    plt.plot(t, v_world[:,main_axis], label=f"v_{axis_name}")
    plt.plot(t, x_world[:,main_axis], label=f"x_{axis_name}")
    plt.legend(); plt.title("Pendulum main-axis motion")
    plt.xlabel("Time [s]"); plt.grid(True, alpha=0.3)
    _savefig(outdir, "pendulum_main_axis")


# ====================================================
# Elevator & square unchanged
# ====================================================
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


# ====================================================
# Keep the before-plots unchanged
# ====================================================
def save_before_plots(outdir, df):
    from src.core import integrate_naive, gyro_tracks_raw
    t = df["t_sec"].to_numpy()
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

    plot_xyz(t, a_raw, "Accelerometer (raw, body frame)", "a (m/s²)", "acc_before_body")
    v_raw, x_raw = integrate_naive(t, a_raw)
    plot_xyz(t, v_raw, "Velocity (raw, naive integration, body frame)", "v (m/s)", "vel_before_body")
    plot_xyz(t, x_raw, "Displacement (raw, naive integration, body frame)", "x (m)", "pos_before_body")
    gyro_dps_raw, gyro_ang_deg_raw = gyro_tracks_raw(df)
    plot_xyz(t, gyro_dps_raw, "Gyroscope Angular Velocity (raw)", "ω (deg/s)", "gyro_VT_before_deg_per_s")
    plot_xyz(t, gyro_ang_deg_raw, "Gyroscope Angle (raw, integrated)", "θ (deg)", "gyro_XT_before_deg")

# ====================================================
# 保留 action 註冊機制入口
# ====================================================
def save_action_plots(outdir, action, df, fs, ori, a_world, v_world, x_world):
    fn = _ACTIONS.get(action)
    if fn:
        fn(outdir, df, fs, ori, a_world, v_world, x_world)