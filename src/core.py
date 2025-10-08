# src/core.py
import numpy as np
import pandas as pd

# -------------------- Helpers --------------------
def _magnitude3(arr: np.ndarray) -> np.ndarray:
    return np.sqrt((arr**2).sum(axis=1))

def _skew(w):
    wx,wy,wz = w
    return np.array([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]], float)

def _exp_so3(w, dt):
    th = np.linalg.norm(w)*dt
    if th < 1e-8: 
        return np.eye(3)
    k = w / (np.linalg.norm(w)+1e-12)
    K = _skew(k)
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)

def _acc_only_to_R(acc):
    """Use accelerometer only to get roll/pitch (yaw unresolved)."""
    g = acc / (np.linalg.norm(acc)+1e-12)
    z_w = -g                 # world z up
    ref = np.array([1,0,0], float)
    if abs(np.dot(ref, z_w)) > 0.9:
        ref = np.array([0,1,0], float)
    x_w = ref - np.dot(ref, z_w)*z_w
    if np.linalg.norm(x_w) < 1e-6:
        x_w = np.array([1,0,0], float)
    else:
        x_w = x_w/np.linalg.norm(x_w)
    y_w = np.cross(z_w, x_w)
    R = np.stack([x_w, y_w, z_w], axis=1)
    return R

def _acc_mag_to_R(acc, mag):
    """Use accelerometer + magnetometer to determine roll/pitch/yaw."""
    g = acc / (np.linalg.norm(acc)+1e-12)
    m = mag / (np.linalg.norm(mag)+1e-12)
    z_w = -g
    m_h = m - np.dot(m, z_w)*z_w
    if np.linalg.norm(m_h) < 1e-6:
        x_w = np.array([1,0,0.0])
    else:
        x_w = m_h / np.linalg.norm(m_h)
    y_w = np.cross(z_w, x_w)
    R = np.stack([x_w, y_w, z_w], axis=1)
    return R

# -------------------- Stationary & Bias --------------------
def detect_stationary(df: pd.DataFrame, fs: float, gyro_thresh_dps=3.0, min_stationary_sec=0.5):
    g = df[["gyro_x","gyro_y","gyro_z"]].to_numpy()    # assume rad/s
    g_dps = np.rad2deg(g)                              # convert to deg/s for threshold
    mag = _magnitude3(g_dps)
    stat = mag < gyro_thresh_dps

    # enforce minimal run length
    min_len = int(min_stationary_sec*fs)
    if min_len > 1:
        run = 0
        for i in range(len(stat)):
            if stat[i]: run += 1
            else:
                if 0 < run < min_len:
                    stat[i-run:i] = False
                run = 0
        if 0 < run < min_len:
            stat[len(stat)-run:len(stat)] = False
    return stat

def estimate_biases(df: pd.DataFrame, fs: float, gyro_thresh_dps=3.0, min_stationary_sec=0.5):
    stat = detect_stationary(df, fs, gyro_thresh_dps, min_stationary_sec)
    if not stat.any():
        n = max(1, int(1.0*fs))
        stat = np.zeros(len(df), bool); stat[:n] = True

    acc_bias  = df.loc[stat, ["acc_x","acc_y","acc_z"]].mean().to_numpy()
    gyro_bias = df.loc[stat, ["gyro_x","gyro_y","gyro_z"]].mean().to_numpy()
    mag_bias  = df.loc[stat, ["mag_x","mag_y","mag_z"]].mean().to_numpy() if {"mag_x","mag_y","mag_z"}.issubset(df.columns) else np.zeros(3)

    return {"acc": acc_bias, "gyro": gyro_bias, "mag": mag_bias}, stat

# Public helper: bias-corrected body-frame accelerometer & gyroscope
def acc_body_bias_corrected(df: pd.DataFrame, biases):
    return df[["acc_x","acc_y","acc_z"]].to_numpy() - biases["acc"]

def gyro_bias_corrected(df: pd.DataFrame, biases):
    return df[["gyro_x","gyro_y","gyro_z"]].to_numpy() - biases["gyro"]  # rad/s

# -------------------- Orientation --------------------
def estimate_orientation(df: pd.DataFrame, fs: float, biases, alpha: float=0.02, use_mag: bool=False):
    t = df["t_sec"].to_numpy()
    acc = acc_body_bias_corrected(df, biases)
    gyro= gyro_bias_corrected(df, biases)
    if {"mag_x","mag_y","mag_z"}.issubset(df.columns):
        mag = df[["mag_x","mag_y","mag_z"]].to_numpy() - biases["mag"]
    else:
        mag = np.zeros_like(acc)

    N = len(df)
    R_bw = np.zeros((N,3,3))
    # init with acc(/mag)
    R = (_acc_mag_to_R(acc[0], mag[0]) if use_mag else _acc_only_to_R(acc[0]))

    for i in range(N):
        if i > 0:
            dt = t[i]-t[i-1]
            R = R @ _exp_so3(gyro[i-1], dt)
        R_am = (_acc_mag_to_R(acc[i], mag[i]) if use_mag else _acc_only_to_R(acc[i]))
        R = (1-alpha)*R + alpha*R_am
        # orthonormalize
        u,_,vt = np.linalg.svd(R)
        R = u@vt
        R_bw[i] = R

    # Euler (roll, pitch, yaw)
    euler = np.zeros((N,3))
    for i in range(N):
        R = R_bw[i]
        pitch = -np.arcsin(np.clip(R[2,0], -1, 1))
        roll  = np.arctan2(R[2,1], R[2,2])
        yaw   = np.arctan2(R[1,0], R[0,0])
        euler[i] = np.array([roll, pitch, yaw])

    return {"R_bw": R_bw, "euler": euler}

# -------------------- World Acc / Integrations --------------------
G = np.array([0,0,9.80665])

def to_world_linear_acc(df: pd.DataFrame, ori, biases):
    acc_b = acc_body_bias_corrected(df, biases)
    R_bw = ori["R_bw"]
    N = len(df)
    a_w = np.zeros_like(acc_b)
    for i in range(N):
        a_w[i] = R_bw[i] @ acc_b[i]
    a_lin = a_w - G  # subtract gravity
    return a_lin

def integrate_velocity_position(t: np.ndarray, a: np.ndarray, stationary_mask: np.ndarray):
    """Generic integrator: a can be body-frame (uncorrected) or world-frame (corrected).
       Applies ZUPT on stationary_mask and per-segment linear detrend."""
    # integrate acceleration to velocity
    v = np.zeros_like(a)
    for i in range(1, len(t)):
        dt = t[i]-t[i-1]
        v[i] = v[i-1] + 0.5*(a[i]+a[i-1])*dt
        if stationary_mask[i]:
            v[i] = 0.0

    # linear detrend per moving segment
    v_dc = v.copy()
    n = len(t); i = 0
    while i < n:
        if not stationary_mask[i]:
            j = i
            while j < n and not stationary_mask[j]:
                j += 1
            ti = t[i:j]
            for d in range(3):
                xi = v_dc[i:j, d]
                if len(ti) > 2:
                    t0 = ti - ti.mean()
                    a_lin = np.dot(t0, xi) / (np.dot(t0, t0) + 1e-12)
                    b = xi.mean() - a_lin * ti.mean()
                    v_dc[i:j, d] = xi - (a_lin*ti + b)
            i = j
        else:
            i += 1

    # integrate velocity to position
    x = np.zeros_like(v_dc)
    for k in range(1, len(t)):
        dt = t[k]-t[k-1]
        x[k] = x[k-1] + 0.5*(v_dc[k]+v_dc[k-1])*dt

    # detrend position per moving segment
    x_dc = x.copy()
    i = 0
    while i < n:
        if not stationary_mask[i]:
            j = i
            while j < n and not stationary_mask[j]:
                j += 1
            ti = t[i:j]
            for d in range(3):
                xi = x_dc[i:j, d]
                if len(ti) > 2:
                    t0 = ti - ti.mean()
                    a_lin = np.dot(t0, xi) / (np.dot(t0, t0) + 1e-12)
                    b = xi.mean() - a_lin * ti.mean()
                    x_dc[i:j, d] = xi - (a_lin*ti + b)
            i = j
        else:
            i += 1

    return v_dc, x_dc

# -------------------- Gyroscope tracks --------------------
def gyro_tracks(df: pd.DataFrame, biases):
    """Return angular velocity [deg/s] and integrated angle [deg] for each axis."""
    t = df["t_sec"].to_numpy()
    w = gyro_bias_corrected(df, biases)             # rad/s
    w_dps = np.rad2deg(w)                           # deg/s
    ang = np.zeros_like(w)                          # rad
    for i in range(1, len(t)):
        dt = t[i]-t[i-1]
        ang[i] = ang[i-1] + 0.5*(w[i]+w[i-1])*dt
    ang_deg = np.rad2deg(ang)
    return w_dps, ang_deg
