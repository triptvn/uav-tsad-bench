"""Convert the three flight datasets to one format: one parquet file per flight.

Columns: t (seconds, uniform grid), channels (float32), label (0/1 per time step), in_air (0/1).
Each dataset also gets meta.csv with flight_id, group, fault_type, n, dur_s, onset_s, end_s.

Resampling uses a zero-order hold (latest sample at or before each tick), so no future value leaks in.
RflyMAD is stored at 50 Hz and decimated to 10 Hz when loaded; ALFA and UAV-SEAD are stored at 10 Hz.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def zoh(t_src, v_src, t_grid):
    """Latest value with timestamp <= t; NaN before the first sample."""
    t_src = np.asarray(t_src, dtype=np.float64)
    order = np.argsort(t_src, kind="stable")
    t_src, v_src = t_src[order], np.asarray(v_src, dtype=np.float64)[order]
    idx = np.searchsorted(t_src, t_grid, side="right") - 1
    out = np.full(len(t_grid), np.nan)
    ok = idx >= 0
    out[ok] = v_src[idx[ok]]
    return out


def quat_to_euler(q0, q1, q2, q3):
    """PX4 quaternion (w, x, y, z) -> roll, pitch, yaw in rad."""
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
    pitch = np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1, 1))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
    return roll, pitch, yaw


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def finalize(df):
    """Back-fill leading NaNs with the first valid value; channels that are entirely missing become 0."""
    ch = [c for c in df.columns if c not in ("t", "label", "in_air")]
    empty = [c for c in ch if df[c].isna().all()]
    df[ch] = df[ch].bfill().fillna(0.0).astype(np.float32)
    df["label"] = df["label"].astype(np.int8)
    if "in_air" in df:
        df["in_air"] = df["in_air"].astype(np.int8)
    return df, empty


# ---------------------------------------------------------------- ULog (RflyMAD, UAV-SEAD)

def _ulog_topics(ulog):
    out = {}
    for d in ulog.data_list:
        if d.multi_id == 0:
            # timestamps are uint64; a few samples precede start_timestamp, so subtracting would wrap around
            d.data["timestamp"] = d.data["timestamp"].astype(np.int64)
            out[d.name] = d.data
    return out


def _ulog_air_segment(tp, t0):
    """(first take-off, last landing) from vehicle_land_detected.landed, or None.

    Using the first touchdown instead would drop the main flight in logs that start with a short test hop.
    """
    ld = tp.get("vehicle_land_detected")
    if ld is None or len(ld["timestamp"]) < 2:
        return None
    t = (ld["timestamp"] - t0) / 1e6
    landed = ld["landed"].astype(int)
    air = np.flatnonzero(landed == 0)
    if len(air) == 0:
        return None
    t_on = t[air[0]]
    after = np.flatnonzero((landed == 1) & (np.arange(len(t)) > air[-1]))
    t_off = t[after[0]] if len(after) else t[-1]
    return t_on, t_off


def _in_air(tp, t0, grid):
    ld = tp.get("vehicle_land_detected")
    if ld is None:
        return np.ones(len(grid))
    v = zoh((ld["timestamp"] - t0) / 1e6, (ld["landed"] == 0).astype(float), grid)
    return np.nan_to_num(v, nan=1.0)


def _add(cols, tp, topic, field, name, t0, grid, scale=1.0):
    d = tp.get(topic)
    if d is None or field not in d:
        cols[name] = np.full(len(grid), np.nan)
        return
    cols[name] = zoh((d["timestamp"] - t0) / 1e6, d[field] * scale, grid)


def _att_block(cols, tp, t0, grid, prefix=""):
    """Measured Euler angles and their setpoint tracking errors."""
    att = tp.get("vehicle_attitude")
    sp = tp.get("vehicle_attitude_setpoint")
    if att is not None and "q[0]" in att:
        r, p, y = quat_to_euler(att["q[0]"], att["q[1]"], att["q[2]"], att["q[3]"])
        ta = (att["timestamp"] - t0) / 1e6
        cols["roll"], cols["pitch"] = zoh(ta, r, grid), zoh(ta, p, grid)
        cols["yaw"] = zoh(ta, np.unwrap(y), grid)
        yaw_w = zoh(ta, y, grid)
    else:
        for k in ("roll", "pitch", "yaw"):
            cols[k] = np.full(len(grid), np.nan)
        yaw_w = cols["yaw"]
    if sp is not None and "q_d[0]" in sp:
        r, p, y = quat_to_euler(sp["q_d[0]"], sp["q_d[1]"], sp["q_d[2]"], sp["q_d[3]"])
        ts = (sp["timestamp"] - t0) / 1e6
        cols["err_roll"] = zoh(ts, r, grid) - cols["roll"]
        cols["err_pitch"] = zoh(ts, p, grid) - cols["pitch"]
        cols["err_yaw"] = wrap(zoh(ts, y, grid) - yaw_w)
    elif sp is not None and "roll_body" in sp:  # older PX4 releases
        ts = (sp["timestamp"] - t0) / 1e6
        cols["err_roll"] = zoh(ts, sp["roll_body"], grid) - cols["roll"]
        cols["err_pitch"] = zoh(ts, sp["pitch_body"], grid) - cols["pitch"]
        cols["err_yaw"] = wrap(zoh(ts, sp["yaw_body"], grid) - yaw_w)
    else:
        for k in ("err_roll", "err_pitch", "err_yaw"):
            cols[k] = np.full(len(grid), np.nan)


def _common_ulog_channels(tp, t0, grid):
    """Channels shared by RflyMAD and UAV-SEAD. The first 12 form channel set K (as in the RflyMAD paper)."""
    c = {}
    av = tp.get("vehicle_angular_velocity")
    if av is not None and "xyz[0]" in av:
        for i, k in enumerate(("p", "q", "r")):
            _add(c, tp, "vehicle_angular_velocity", f"xyz[{i}]", k, t0, grid)
    else:  # older PX4 releases log body rates in vehicle_attitude
        for f, k in (("rollspeed", "p"), ("pitchspeed", "q"), ("yawspeed", "r")):
            _add(c, tp, "vehicle_attitude", f, k, t0, grid)
    for i, k in enumerate(("ax", "ay", "az")):
        _add(c, tp, "sensor_combined", f"accelerometer_m_s2[{i}]", k, t0, grid)
    _att_block(c, tp, t0, grid)
    for f in ("vx", "vy", "vz"):
        _add(c, tp, "vehicle_local_position", f, f, t0, grid)
    # channel set F adds the following
    for f in ("x", "y", "z"):
        _add(c, tp, "vehicle_local_position", f, "pos_" + f, t0, grid)
    for i, k in enumerate(("gx", "gy", "gz")):
        _add(c, tp, "sensor_combined", f"gyro_rad[{i}]", k, t0, grid)
    rs = tp.get("vehicle_rates_setpoint")
    for f, k, m in (("roll", "err_p", "p"), ("pitch", "err_q", "q"), ("yaw", "err_r", "r")):
        if rs is not None and f in rs:
            c[k] = zoh((rs["timestamp"] - t0) / 1e6, rs[f], grid) - c[m]
        else:
            c[k] = np.full(len(grid), np.nan)
    for i in range(4):
        _add(c, tp, "actuator_outputs", f"output[{i}]", f"motor{i}", t0, grid)
    for i, k in enumerate(("u_roll", "u_pitch", "u_yaw", "u_thrust")):
        _add(c, tp, "actuator_controls_0", f"control[{i}]", k, t0, grid)
    _add(c, tp, "battery_status", "voltage_v", "batt_v", t0, grid)
    _add(c, tp, "distance_sensor", "current_distance", "range", t0, grid)
    return c


def read_rflymad_case(ulg_path, hz=50.0):
    """One RflyMAD flight. Label = intervals where rfly_ctrl_lxl.id differs from its idle value (1500)."""
    from pyulog import ULog
    u = ULog(str(ulg_path))
    tp = _ulog_topics(u)
    t0 = int(u.start_timestamp)
    seg = _ulog_air_segment(tp, t0)
    t_end = (int(u.last_timestamp) - t0) / 1e6
    t_on, t_off = seg if seg else (0.0, t_end)
    grid = np.arange(np.ceil(t_on * hz) / hz, t_off, 1.0 / hz)
    c = _common_ulog_channels(tp, t0, grid)
    for i, k in enumerate(("mx", "my", "mz")):
        _add(c, tp, "sensor_mag", ["x", "y", "z"][i], k, t0, grid)
    _add(c, tp, "vehicle_air_data", "baro_alt_meter", "baro_alt", t0, grid)
    for f, k in (("vel_n_m_s", "gps_vn"), ("vel_e_m_s", "gps_ve"), ("vel_d_m_s", "gps_vd"), ("eph", "gps_eph")):
        _add(c, tp, "vehicle_gps_position", f, k, t0, grid)
    fl = tp["rfly_ctrl_lxl"]
    tf = (fl["timestamp"] - t0) / 1e6
    fid = fl["id"].astype(np.int64)
    base = 1500
    active = (fid != base) & (fid != 0)
    lab = zoh(tf, active.astype(float), grid)
    lab = np.nan_to_num(lab, nan=0.0)
    ids = sorted(set(fid[active].tolist()))
    on = tf[np.flatnonzero(active)[0]] if active.any() else np.nan
    off = tf[np.flatnonzero(active)[-1]] if active.any() else np.nan
    df = pd.DataFrame({"t": grid - grid[0], **c, "label": lab, "in_air": _in_air(tp, t0, grid)})
    df, empty = finalize(df)
    info = dict(n=len(df), dur_s=float(grid[-1] - grid[0]), onset_s=float(on - grid[0]) if active.any() else np.nan,
                end_s=float(off - grid[0]) if active.any() else np.nan, fault_ids=ids,
                air_seg=bool(seg), empty=empty, id_seq=_runs(fid))
    return df, info


def _runs(a):
    """Sequence of distinct consecutive values, for eyeballing the fault schedule."""
    a = np.asarray(a)
    keep = np.r_[True, a[1:] != a[:-1]]
    return a[keep].tolist()[:12]


def read_uavsead_log(ulg_path, anns, hz=10.0):
    """One UAV-SEAD log. Label = union of all annotated ranges (ULog microseconds) of all annotations."""
    from pyulog import ULog
    u = ULog(str(ulg_path))
    tp = _ulog_topics(u)
    t0 = int(u.start_timestamp)
    seg = _ulog_air_segment(tp, t0)
    t_end = (int(u.last_timestamp) - t0) / 1e6
    t_on, t_off = seg if seg else (0.0, t_end)
    grid = np.arange(np.ceil(t_on * hz) / hz, t_off, 1.0 / hz)
    c = _common_ulog_channels(tp, t0, grid)
    ei = tp.get("ekf2_innovations")
    for i in range(6):
        if ei is not None and f"vel_pos_innov[{i}]" in ei:
            _add(c, tp, "ekf2_innovations", f"vel_pos_innov[{i}]", f"innov{i}", t0, grid)
        else:
            c[f"innov{i}"] = np.full(len(grid), np.nan)
    _add(c, tp, "ekf2_innovations", "hagl_innov", "innov_hagl", t0, grid)
    for f in ("pos_test_ratio", "vel_test_ratio", "hgt_test_ratio", "mag_test_ratio"):
        _add(c, tp, "estimator_status", f, f, t0, grid)
    lab = np.zeros(len(grid))
    for _sig, ranges in [r for ann in anns for r in ann["ranges"]]:
        for a, b in ranges:
            a, b = (min(a, b) - t0) / 1e6, (max(a, b) - t0) / 1e6
            lab[(grid >= a) & (grid <= b)] = 1.0
    df = pd.DataFrame({"t": grid - grid[0], **c, "label": lab, "in_air": _in_air(tp, t0, grid)})
    df, empty = finalize(df)
    pos = np.flatnonzero(df["label"].values)
    info = dict(n=len(df), dur_s=float(grid[-1] - grid[0]),
                onset_s=float(df["t"].values[pos[0]]) if len(pos) else np.nan,
                end_s=float(df["t"].values[pos[-1]]) if len(pos) else np.nan,
                air_seg=bool(seg), empty=empty, n_ranges=sum(len(r) for ann in anns for _, r in ann["ranges"]))
    return df, info


# ---------------------------------------------------------------- ALFA (processed CSV release)

ALFA_CH = [  # (topic, column, channel); imu-data_raw is logged at 10 Hz, imu-data only at about 2.5 Hz
    ("mavros-imu-data_raw", "field.angular_velocity.x", "p"),
    ("mavros-imu-data_raw", "field.angular_velocity.y", "q"),
    ("mavros-imu-data_raw", "field.angular_velocity.z", "r"),
    ("mavros-imu-data_raw", "field.linear_acceleration.x", "ax"),
    ("mavros-imu-data_raw", "field.linear_acceleration.y", "ay"),
    ("mavros-imu-data_raw", "field.linear_acceleration.z", "az"),
    ("mavros-nav_info-roll", "field.measured", "roll"),
    ("mavros-nav_info-pitch", "field.measured", "pitch"),
    ("mavros-nav_info-yaw", "field.measured", "yaw"),
    ("mavros-local_position-velocity", "field.twist.linear.x", "vx"),
    ("mavros-local_position-velocity", "field.twist.linear.y", "vy"),
    ("mavros-local_position-velocity", "field.twist.linear.z", "vz"),
    ("mavros-nav_info-roll", "field.commanded", "_cmd_roll"),
    ("mavros-nav_info-pitch", "field.commanded", "_cmd_pitch"),
    ("mavros-nav_info-yaw", "field.commanded", "_cmd_yaw"),
    ("mavros-nav_info-airspeed", "field.measured", "airspeed"),
    ("mavros-nav_info-airspeed", "field.commanded", "_cmd_airspeed"),
    ("mavros-nav_info-velocity", "field.meas_x", "nav_vx"),
    ("mavros-nav_info-velocity", "field.meas_y", "nav_vy"),
    ("mavros-nav_info-velocity", "field.meas_z", "nav_vz"),
    ("mavros-vfr_hud", "field.throttle", "throttle"),
    ("mavros-vfr_hud", "field.climb", "climb"),
    ("mavros-global_position-rel_alt", "field.data", "rel_alt"),
    ("mavros-rc-out", "field.channels0", "rc0"),
    ("mavros-rc-out", "field.channels1", "rc1"),
    ("mavros-rc-out", "field.channels2", "rc2"),
    ("mavros-rc-out", "field.channels3", "rc3"),
]


def alfa_group(name):
    """Sequences _1, _2 cut from the same bag share a group."""
    m = re.match(r"carbonZ_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", name)
    return m.group(1)


def alfa_fault(name):
    s = re.sub(r"^carbonZ_[\d-]+_", "", name)
    s = re.sub(r"^\d_", "", s)
    return s.replace("_with_emr_traj", "")


def read_alfa_seq(folder, hz=10.0):
    folder = Path(folder)
    name = folder.name
    tabs, t_min, t_max = {}, [], []
    for topic in sorted({t for t, _, _ in ALFA_CH}):
        f = folder / f"{name}-{topic}.csv"
        if f.exists():
            d = pd.read_csv(f)
            tabs[topic] = d
            t_min.append(d["%time"].iloc[0])
            t_max.append(d["%time"].iloc[-1])
    t0 = max(min(t_min), 0)
    t_on, t_off = (max(t_min) - t0) / 1e9, (min(t_max) - t0) / 1e9  # span covered by every topic
    grid = np.arange(np.ceil(t_on * hz) / hz, t_off, 1.0 / hz)
    c = {}
    for topic, col, ch in ALFA_CH:
        d = tabs.get(topic)
        c[ch] = zoh((d["%time"].values - t0) / 1e9, d[col].values, grid) if d is not None and col in d else np.full(len(grid), np.nan)
    for k in ("roll", "pitch", "yaw", "airspeed"):
        c["err_" + k] = c.pop("_cmd_" + k) - c[k]
    c["err_yaw"] = (c["err_yaw"] + 180) % 360 - 180  # nav_info angles are in degrees
    ok = ~np.isnan(c["yaw"])
    c["yaw"][ok] = np.degrees(np.unwrap(np.radians(c["yaw"][ok])))
    onset = np.nan
    for f in folder.glob(f"{name}-failure_status-*.csv"):
        d = pd.read_csv(f)
        if len(d):
            tt = (d["%time"].iloc[0] - t0) / 1e9
            onset = tt if np.isnan(onset) else min(onset, tt)
    lab = (grid >= onset).astype(float) if not np.isnan(onset) else np.zeros(len(grid))
    df = pd.DataFrame({"t": grid - grid[0], **c, "label": lab})
    df, empty = finalize(df)
    info = dict(n=len(df), dur_s=float(grid[-1] - grid[0]),
                onset_s=float(onset - grid[0]) if not np.isnan(onset) else np.nan,
                end_s=float(grid[-1] - grid[0]) if not np.isnan(onset) else np.nan, empty=empty,
                abs_t0=float(t0 / 1e9 + grid[0]), abs_t1=float(t0 / 1e9 + grid[-1]))
    return df, info


def overlap_groups(keys, t0s, t1s):
    """Merge sequences that share a bag or overlap in wall-clock time (several ALFA sequences cover the same flight)."""
    n = len(keys)
    par = list(range(n))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if keys[i] == keys[j] or (t1s[i] > t0s[j] and t1s[j] > t0s[i]):
                par[find(j)] = find(i)
    roots = [find(i) for i in range(n)]
    name = {r: f"g{k:03d}" for k, r in enumerate(dict.fromkeys(roots))}
    return [name[r] for r in roots]


def save(df, out_dir, flight_id):
    out_dir = Path(out_dir) / "flights"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = flight_id.replace("/", "__")
    df.to_parquet(out_dir / f"{safe}.parquet", index=False)
    return safe


def write_meta(rows, out_dir):
    m = pd.DataFrame(rows)
    for c in m.columns:
        if m[c].apply(lambda v: isinstance(v, (list, dict))).any():
            m[c] = m[c].apply(json.dumps)
    m.to_csv(Path(out_dir) / "meta.csv", index=False)
    return m
