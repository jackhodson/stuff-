"""
model_trainer.py  v18 — Physics-Formula Stuff+
────────────────────────────────────────────────
Direct physics-formula scorer with domain-knowledge weights, calibrated from
actual training data (league means/stds per pitch type).

Dimensions scored per pitch: velocity, iVB (arm-angle adjusted), HB (arm-angle
adjusted), VAA (arm-angle adjusted), extension, spin rate.
Gyro sliders get a separate scoring path (velocity + vertical split + VAA diff
+ low spin efficiency), with MAX(normal, gyro) taken per pitch.

Rank normalization → proper 100-mean distribution.
Hard velocity floor for slow fastballs.

FIXES (v18):
  - VAA now computed via correct Statcast kinematic formula:
      vy_f = -sqrt(vy0² - 2*ay*(y0 - yf))   where y0=50, yf=17/12
      t    = (vy_f - vy0) / ay
      vz_f = vz0 + az*t
      VAA  = -arctan(vz_f / vy_f) * (180/pi)
  - HB sign convention: POSITIVE = arm side for BOTH handedness.
      RHP pfx_x is positive arm-side already.
      LHP pfx_x is negative arm-side in Statcast → flip sign so positive = arm side.
  - Arm angle on movement plot now correctly derived from release position.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats

MODEL_DIR       = Path("data/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
NORM_PATH       = MODEL_DIR / "stuff_plus_norm.pkl"
VAA_PATH        = MODEL_DIR / "vaa_coeffs.pkl"
AUX_PATH        = MODEL_DIR / "aux_params.pkl"
MODEL_PATH      = MODEL_DIR / "stuff_plus_model.pkl"   # sentinel only
PHYSICS_VERSION = "v18_sinker_hbrun"

FB_TYPES       = {"FF", "SI"}
FC_TYPES       = {"FC"}
BREAKING_TYPES = {"SL", "ST", "SW", "CU", "KC"}
OFFSPEED_TYPES = {"CH", "FS", "FO"}
GYRO_SL_TYPES  = {"SL", "ST", "SW"}

# ─── DIMENSION WEIGHTS ────────────────────────────────────────────────────────
WEIGHTS = {
    "FF": {"velo":2.5, "ivb":1.8, "hb":0.6, "vaa":1.5, "ext":0.8, "spin":0.8},
    "SI": {"velo":2.0, "si_depth":2.0, "hb_run":2.5, "vaa":0.5, "ext":0.5, "spin":0.4},
    "FC": {"velo":2.2, "ivb":1.2, "hb":0.8, "vaa":0.8, "ext":0.5, "spin":0.6},
    "CH": {"tunnel":3.0, "velo_gap":1.5, "hb_tunnel":2.0},
    "FS": {"tunnel":2.5, "velo_gap":2.0, "hb_tunnel":1.5},
    "FO": {"tunnel":2.5, "velo_gap":1.5, "hb_tunnel":1.5},
    "CU": {"velo":2.0, "ivb":2.2, "hb":1.0, "vaa":0.8, "spin":1.0},
    "KC": {"velo":2.0, "ivb":2.2, "hb":1.2, "vaa":0.8, "spin":1.0},
    "SL": {"velo":1.8, "ivb":1.2, "hb":2.2, "haa":0.8, "vaa":0.5, "spin":0.8},
    "ST": {"velo":1.8, "ivb":0.8, "hb":2.8, "haa":1.5, "vaa":0.3, "spin":0.8},
    "SW": {"velo":1.8, "ivb":0.8, "hb":2.8, "haa":1.5, "vaa":0.3, "spin":0.8},
    "SV": {"velo":1.8, "ivb":1.5, "hb":2.0, "vaa":0.5, "spin":0.8},
}

GYRO_WEIGHTS = {
    "velo":    2.0,
    "vsplit":  3.0,
    "vaadiff": 1.5,
    "spineff": 2.0,
}
GYRO_IVB_THRESHOLD = 4.0
GYRO_HB_THRESHOLD  = 5.0
GYRO_VSPLIT_MEAN   = 10.0
GYRO_VSPLIT_STD    = 3.0
GYRO_VAADIFF_MEAN  = 2.0
GYRO_VAADIFF_STD   = 1.0
GYRO_SPINEFF_MEAN  = 0.35
GYRO_SPINEFF_STD   = 0.08

HIGH_IVB_SW_IVB_THRESHOLD = 5.0
HIGH_IVB_SW_HB_MIN        = 8.0
HIGH_IVB_SW_COMBINED_MEAN = 18.0
HIGH_IVB_SW_COMBINED_STD  = 3.0
HIGH_IVB_SW_WEIGHTS = {
    "velo":     1.5,
    "combined": 3.0,
    "balance":  2.5,
}

OFFSPEED_DEPTH_GAP_MEAN = 9.0
OFFSPEED_DEPTH_GAP_STD  = 2.5
OFFSPEED_VELO_GAP_MEAN  = 9.0
OFFSPEED_VELO_GAP_STD   = 2.5
OFFSPEED_HB_GAP_MEAN    = 2.0
OFFSPEED_HB_GAP_STD     = 3.5

FB_VELO_BASELINE = 93.0
FB_VELO_RATE     = 2.5


# ─── STATCAST VAA / HAA  (correct kinematic formula per Statcast docs) ────────
#
#   vy_f = -sqrt(vy0² - 2 * ay * (y0 - yf))
#   t    = (vy_f - vy0) / ay
#   vz_f = vz0 + az * t
#   vx_f = vx0 + ax * t
#   VAA  = -arctan(vz_f / vy_f) * (180 / pi)
#   HAA  = -arctan(vx_f / vy_f) * (180 / pi)
#
# y0 = 50 ft (Statcast measurement plane), yf = 17/12 ft (front of home plate)

def compute_vaa_haa(df: pd.DataFrame):
    try:
        vy0 = df["vy0"].values.astype(float)
        vz0 = df["vz0"].values.astype(float)
        vx0 = df["vx0"].values.astype(float)
        ay  = df["ay"].values.astype(float)
        az  = df["az"].values.astype(float)
        ax  = df["ax"].values.astype(float)

        y0 = 50.0          # feet — Statcast reference plane
        yf = 17.0 / 12.0   # feet — front of home plate

        # Final y-velocity at home plate (kinematic, always negative — toward plate)
        inner = vy0**2 - 2.0 * ay * (y0 - yf)
        # Guard against floating-point negatives under sqrt
        inner = np.clip(inner, 0.0, None)
        vy_f  = -np.sqrt(inner)

        # Time of flight from y=50 to home plate
        # Avoid divide-by-zero where ay ≈ 0
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(np.abs(ay) > 1e-6,
                         (vy_f - vy0) / ay,
                         (y0 - yf) / np.abs(vy0))

        vz_f = vz0 + az * t
        vx_f = vx0 + ax * t

        with np.errstate(invalid="ignore", divide="ignore"):
            vaa = -np.arctan(vz_f / vy_f) * (180.0 / np.pi)
            haa = -np.arctan(vx_f / vy_f) * (180.0 / np.pi)

        vaa = np.where(np.isfinite(vaa), vaa, np.nan)
        haa = np.where(np.isfinite(haa), haa, np.nan)

    except Exception:
        vaa = np.full(len(df), np.nan)
        haa = np.full(len(df), np.nan)

    return vaa, haa


def fit_vaa_adjustment(df):
    ph, v = df["plate_z"].values, df["_vaa"].values
    mask  = np.isfinite(ph) & np.isfinite(v)
    return tuple(np.polyfit(ph[mask], v[mask], 1)) if mask.sum() > 100 \
           else (0.0, float(np.nanmean(v[mask])))


def apply_vaa_adjustment(df, coeffs):
    a, b = coeffs
    df   = df.copy()
    df["adj_vaa"] = df["_vaa"] - (a * df["plate_z"].values + b)
    return df


# ─── HB SIGN CONVENTION ───────────────────────────────────────────────────────
# Statcast pfx_x convention:
#   RHP: positive = catcher's right = pitcher's arm side  ✓ (no flip needed)
#   LHP: positive = catcher's right = pitcher's GLOVE side (needs flip)
#
# We want: positive HB = arm side (toward the pitcher's throwing-arm side)
# so that the same physical movement (e.g. cut) always has the same sign
# regardless of handedness.
#
# After this transform:
#   RHP sinker/4-seam: hb > 0 (arm side, i.e. right)
#   LHP sinker/4-seam: hb > 0 (arm side, i.e. left)
#   Both: curveball hb < 0 (glove side)

def fix_hb_handedness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flip HB sign for LHP so that positive = arm side for both handedness.
    Operates on the 'hb' column (already converted to inches from pfx_x).
    """
    df = df.copy()
    if "p_throws" in df.columns and "hb" in df.columns:
        lhp_mask = df["p_throws"] == "L"
        df.loc[lhp_mask, "hb"] = -df.loc[lhp_mask, "hb"]
    return df


# ─── ARM ANGLE (correct release-point geometry) ───────────────────────────────
# Arm angle = degrees above horizontal from a ~5.0 ft baseline reference,
# matching the scale Statcast publishes on Baseball Savant:
#   ~0-10°  = sidearm
#   ~20-30° = low 3/4
#   ~35-45° = 3/4 / high 3/4
#   ~55-70° = over the top
#
# Formula: arctan2(release_pos_z - 5.0, abs(release_pos_x))
#
# release_pos_x in Statcast is from the CATCHER's perspective:
#   RHP: negative (catcher's left = pitcher's arm side)
#   LHP: positive (catcher's right = pitcher's arm side)
# We take abs(rx) so both handedness use the same geometric formula —
# NO handedness flip is needed here (abs covers it).
# The HB flip (fix_hb_handedness) is separate and only for movement direction.

def compute_arm_angle(df: pd.DataFrame) -> np.ndarray:
    """
    Return arm angle in degrees (0=sidearm, ~60=over-top).
    Matches the scale of Statcast's published arm angle metric.
    No handedness flip required — abs(release_pos_x) handles both RHP and LHP.
    """
    rx = df["release_pos_x"].values.astype(float)
    rz = df["release_pos_z"].values.astype(float)

    # abs(rx) = horizontal distance from center, regardless of handedness
    # 5.0 ft baseline: the approximate height at which arm angle = 0° (horizontal)
    arm_angle = np.degrees(np.arctan2(rz - 5.0, np.abs(rx)))

    # Clip to valid range
    arm_angle = np.clip(arm_angle, 0.0, 90.0)
    return arm_angle


# ─── LEAGUE STATS ─────────────────────────────────────────────────────────────

def fit_league_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for pt, grp in df.groupby("pitch_type"):
        if len(grp) < 100:
            continue
        def ms(col, fm=0.0, fs=1.0):
            v = grp[col].dropna() if col in grp.columns else pd.Series(dtype=float)
            return (float(v.mean()), max(float(v.std()), 0.01)) if len(v) >= 30 \
                   else (fm, fs)
        stats[pt] = {
            "velo": ms("release_speed",    92.0,  2.0),
            "ivb":  ms("ivb",             13.0,  3.5),
            "hb":   ms("hb",               8.0,  3.5),
            "vaa":  ms("adj_vaa",         -4.5,  0.8),
            "ext":  ms("release_extension", 6.2,  0.3),
            "spin": ms("release_spin_rate",2200., 300.),
        }
    return stats


# ─── FASTBALL REFERENCE PER PITCHER ──────────────────────────────────────────

def compute_fb_reference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pitcher_col = next((c for c in ["pitcher","pitcher_id","player_id"]
                        if c in df.columns), None)

    for c in ["fb_ivb","fb_hb","fb_velo","fb_vaa"]:
        df[c] = np.nan

    if pitcher_col is None:
        return df

    fb_df = df[df["pitch_type"].isin({"FF","SI"})]
    if len(fb_df) == 0:
        return df

    ref = (fb_df.groupby(pitcher_col)[["ivb","hb","release_speed","adj_vaa"]]
               .mean()
               .rename(columns={"ivb":"fb_ivb","hb":"fb_hb",
                                 "release_speed":"fb_velo","adj_vaa":"fb_vaa"}))

    pid = df[pitcher_col].values
    for col in ["fb_ivb","fb_hb","fb_velo","fb_vaa"]:
        lut = ref[col].to_dict()
        df[col] = [lut.get(p, np.nan) for p in pid]

    return df


# ─── ARM-ANGLE SHAPE MODELS ───────────────────────────────────────────────────

def fit_arm_angle_models(df: pd.DataFrame) -> dict:
    """
    Fit per-pitch-type linear regressions: expected_dim = a*arm_angle + b.

    Dimensions fitted:
      ivb     — raw induced vertical break (inches)
      hb      — horizontal break, arm-side positive (inches)
      adj_vaa — plate-z-adjusted vertical approach angle (degrees)
      _haa_abs — abs(HAA) — horizontal approach angle magnitude (degrees)
                 Lower arm slots naturally generate higher abs(HAA) for breaking
                 balls; scoring HAA deviations relative to slot expectation
                 prevents sidearm pitchers from getting an unfair HAA bonus.
    """
    models = {}
    if "arm_angle" not in df.columns:
        return models

    # Pre-compute abs HAA column for regression
    df = df.copy()
    if "_haa" in df.columns:
        df["_haa_abs"] = df["_haa"].abs()

    for pt, grp in df.groupby("pitch_type"):
        if len(grp) < 200:
            continue
        aa = grp["arm_angle"].values
        m  = {}
        for col in ["ivb", "hb", "adj_vaa", "_haa_abs"]:
            if col not in grp.columns:
                continue
            y    = grp[col].values
            mask = np.isfinite(aa) & np.isfinite(y)
            m[col] = tuple(np.polyfit(aa[mask], y[mask], 1)) if mask.sum() > 50 \
                     else (0.0, float(np.nanmean(y[mask])))
        models[pt] = m
    return models


def _slot_expected(aa, model):
    a, b = model
    return a * aa + b


# ─── HAA STATS ────────────────────────────────────────────────────────────────

def fit_haa_stats(df: pd.DataFrame) -> dict:
    stats = {}
    if "_haa" not in df.columns:
        return stats
    for pt, grp in df.groupby("pitch_type"):
        v = grp["_haa"].dropna()
        if len(v) < 50:
            continue
        stats[pt] = {"mean": float(v.mean()), "std": max(float(v.std()), 0.01)}
    return stats


# ─── EXPONENTIAL OUTLIER TRANSFORM ────────────────────────────────────────────

def _exp(z, scale=2.5):
    z = np.asarray(z, dtype=float)
    return np.sign(z) * (np.exp(np.abs(z) / scale) - 1.0)


# ─── DIRECT FORMULA SCORER ────────────────────────────────────────────────────

def compute_stuff_raw(df: pd.DataFrame, league_stats: dict,
                      arm_models: dict = None,
                      haa_stats:  dict = None) -> np.ndarray:
    scores   = np.zeros(len(df))
    has_arm  = "arm_angle" in df.columns and arm_models
    has_haa  = "_haa" in df.columns and haa_stats
    has_spin = "release_spin_rate" in df.columns

    for pt, w in WEIGHTS.items():
        mask = (df["pitch_type"] == pt).values
        if not mask.any():
            continue
        lg = league_stats.get(pt)
        if lg is None:
            continue

        rows     = df[mask]
        idx      = np.where(mask)[0]
        s        = np.zeros(mask.sum())
        aa       = rows["arm_angle"].values if has_arm else None
        am       = (arm_models or {}).get(pt, {})
        ivb_vals = rows["ivb"].values
        hb_vals  = rows["hb"].values

        # ── Velocity ────────────────────────────────────────────────────────
        velo_z = (rows["release_speed"].values - lg["velo"][0]) / lg["velo"][1]
        s += w.get("velo", 0) * _exp(velo_z)

        # ── iVB: arm-angle adjusted ──────────────────────────────────────────
        if has_arm and "ivb" in am and aa is not None:
            ivb_dev = ivb_vals - _slot_expected(aa, am["ivb"])
        else:
            ivb_dev = ivb_vals - lg["ivb"][0]
        ivb_z = ivb_dev / lg["ivb"][1]
        if pt not in FB_TYPES | FC_TYPES:
            ivb_z = -ivb_z
        s += w.get("ivb", 0) * _exp(ivb_z)

        # ── HB: arm-angle adjusted ───────────────────────────────────────────
        # hb is now arm-side-positive for both handedness.
        # For FB/FC: more arm-side run (positive) is better.
        # For offspeed/breaking: scored on total deviation from slot.
        if has_arm and "hb" in am and aa is not None:
            hb_dev = hb_vals - _slot_expected(aa, am["hb"])
        else:
            hb_dev = hb_vals - lg["hb"][0]
        hb_z = hb_dev / lg["hb"][1]
        s += w.get("hb", 0) * _exp(hb_z)

        # ── VAA: arm-angle adjusted ──────────────────────────────────────────
        vaa_vals = rows["adj_vaa"].values
        if has_arm and "adj_vaa" in am and aa is not None:
            vaa_dev = vaa_vals - _slot_expected(aa, am["adj_vaa"])
        else:
            vaa_dev = vaa_vals - lg["vaa"][0]
        vaa_z = vaa_dev / lg["vaa"][1]
        if pt not in FB_TYPES | FC_TYPES:
            vaa_z = -vaa_z
        s += w.get("vaa", 0) * _exp(vaa_z)

        # ── Extension ───────────────────────────────────────────────────────
        if "ext" in w:
            ext_z = (rows["release_extension"].values - lg["ext"][0]) / lg["ext"][1]
            s += w["ext"] * _exp(ext_z)

        # ── Spin rate ────────────────────────────────────────────────────────
        if "spin" in w and has_spin:
            spin_raw = pd.to_numeric(rows["release_spin_rate"], errors="coerce").values
            spin_z   = np.where(
                np.isfinite(spin_raw),
                (spin_raw - lg["spin"][0]) / lg["spin"][1],
                0.0
            )
            s += w["spin"] * _exp(spin_z)

        # ── Sinker: arm-side run + depth ───────────────────────────────────
        if "hb_run" in w:
            SI_HB_MEAN = 13.0
            SI_HB_STD  = 2.5
            # hb_vals already arm-side-positive; sinker arm-side run = positive hb
            hb_run_z = (hb_vals - SI_HB_MEAN) / SI_HB_STD
            s += w["hb_run"] * _exp(hb_run_z)

        if "si_depth" in w:
            if has_arm and "ivb" in am and aa is not None:
                expected_ivb = _slot_expected(aa, am["ivb"])
                depth_dev    = expected_ivb - ivb_vals
            else:
                depth_dev    = lg["ivb"][0] - ivb_vals
            SI_DEPTH_STD = 2.5
            depth_z = depth_dev / SI_DEPTH_STD
            s += w["si_depth"] * _exp(depth_z)

            if "hb_run" in w:
                both_elite   = (depth_z > 0.5) & (hb_vals > 16.0)
                outlier_z    = np.minimum(depth_z, (hb_vals - 16.0) / 2.0)
                s += both_elite.astype(float) * 1.0 * np.clip(_exp(outlier_z), 0, 3)

        # ── HAA ─────────────────────────────────────────────────────────────
        # HAA is computed via the correct Statcast kinematic formula (same t as VAA).
        # Typical values (abs): SL ~3-4°, ST/SW ~4-6°, elite sweeper ~6-9°.
        #
        # ARM-ANGLE ADJUSTED: sidearm pitchers naturally generate larger abs(HAA)
        # because of their release geometry, not because of superior horizontal
        # approach quality. We compare abs(HAA) to the slot-expected abs(HAA)
        # for that pitch type using the same linear model pattern as iVB/HB/VAA.
        #
        # haa_dev = abs(_haa) - slot_expected_abs_haa(arm_angle)
        # Positive dev = more horizontal approach than arm slot predicts = good.
        if "haa" in w and has_haa and pt in haa_stats:
            hs      = haa_stats[pt]
            haa_abs = np.abs(rows["_haa"].values)
            # Use arm-angle adjusted baseline if model available
            if has_arm and "_haa_abs" in am and aa is not None:
                expected_haa = _slot_expected(aa, am["_haa_abs"])
                haa_dev      = haa_abs - expected_haa
                haa_z        = haa_dev / max(hs["std"], 0.01)
            else:
                # Fallback: compare to league-average HAA for pitch type
                haa_z = (haa_abs - abs(hs["mean"])) / hs["std"]
            s += w["haa"] * _exp(haa_z)

        # ── CH/FS/FO: graded on deception vs own fastball ──────────────────
        if any(k in w for k in ("tunnel", "velo_gap", "hb_tunnel")):
            fb_ivb_ref  = rows["fb_ivb"].values
            fb_velo_ref = rows["fb_velo"].values
            fb_hb_ref   = rows["fb_hb"].values
            os_velo     = rows["release_speed"].values

            depth_gap  = np.where(np.isfinite(fb_ivb_ref),
                                  fb_ivb_ref - ivb_vals,
                                  OFFSPEED_DEPTH_GAP_MEAN)
            depth_z_os = (depth_gap - OFFSPEED_DEPTH_GAP_MEAN) / OFFSPEED_DEPTH_GAP_STD
            s += w.get("tunnel", 0) * np.clip(_exp(depth_z_os), -3, 3)

            velo_gap_mph = np.where(np.isfinite(fb_velo_ref),
                                    fb_velo_ref - os_velo,
                                    OFFSPEED_VELO_GAP_MEAN)
            velo_gap_z   = (velo_gap_mph - OFFSPEED_VELO_GAP_MEAN) / OFFSPEED_VELO_GAP_STD
            s += w.get("velo_gap", 0) * np.clip(_exp(velo_gap_z), -3, 3)

            # HB tunnel: arm-side separation from own FB
            # Both hb_vals and fb_hb_ref are now arm-side-positive
            hb_gap_raw = np.where(np.isfinite(fb_hb_ref),
                                  hb_vals - fb_hb_ref,
                                  0.0)
            hb_gap_z   = (hb_gap_raw - OFFSPEED_HB_GAP_MEAN) / OFFSPEED_HB_GAP_STD
            s += w.get("hb_tunnel", 0) * np.clip(_exp(hb_gap_z), -3, 3)

        # ── High-iVB sweeper ────────────────────────────────────────────────
        if pt in {"ST", "SW"}:
            if has_arm and "ivb" in am and aa is not None:
                expected_ivb_sw = _slot_expected(aa, am["ivb"])
                ivb_above_expected = ivb_vals - expected_ivb_sw
                hi_ivb_mask = ivb_above_expected > HIGH_IVB_SW_IVB_THRESHOLD
                hb_enough   = (np.abs(hb_vals - _slot_expected(aa, am.get("hb", (0,1)))) > HIGH_IVB_SW_HB_MIN
                               if "hb" in am else np.abs(hb_vals) > HIGH_IVB_SW_HB_MIN)
            else:
                hi_ivb_mask = ivb_vals > 3.0
                hb_enough   = np.abs(hb_vals) > HIGH_IVB_SW_HB_MIN

            hi_sw_mask = hi_ivb_mask & hb_enough
            if hi_sw_mask.any():
                hi_idx = np.where(hi_sw_mask)[0]
                hr     = rows.iloc[hi_idx]

                hi_velo_z = (hr["release_speed"].values - lg["velo"][0]) / lg["velo"][1]
                hi_s      = HIGH_IVB_SW_WEIGHTS["velo"] * _exp(hi_velo_z)

                combined  = np.abs(hr["ivb"].values) + np.abs(hr["hb"].values)
                comb_z    = (combined - HIGH_IVB_SW_COMBINED_MEAN) / HIGH_IVB_SW_COMBINED_STD
                hi_s     += HIGH_IVB_SW_WEIGHTS["combined"] * np.clip(_exp(comb_z), -3, 3)

                imbalance   = np.abs(np.abs(hr["ivb"].values) - np.abs(hr["hb"].values))
                balance_z   = -(imbalance - 3.0) / 3.0
                hi_s       += HIGH_IVB_SW_WEIGHTS["balance"] * np.clip(_exp(balance_z), -3, 3)

                s[hi_idx] = np.maximum(s[hi_idx], hi_s)

        # ── Gyro slider ─────────────────────────────────────────────────────
        if pt in GYRO_SL_TYPES:
            if has_arm and "ivb" in am and "hb" in am and aa is not None:
                gyro_mask = (
                    np.abs(ivb_vals - _slot_expected(aa, am["ivb"])) < GYRO_IVB_THRESHOLD
                ) & (
                    np.abs(hb_vals  - _slot_expected(aa, am["hb"])) < GYRO_HB_THRESHOLD
                )
            else:
                gyro_mask = (np.abs(ivb_vals) < 5.0) & (np.abs(hb_vals) < 5.0)

            if gyro_mask.any():
                gi    = np.where(gyro_mask)[0]
                gr    = rows.iloc[gi]

                g_velo_z = (gr["release_speed"].values - lg["velo"][0]) / lg["velo"][1]
                g_s      = GYRO_WEIGHTS["velo"] * _exp(g_velo_z)

                fb_ivb_g = gr["fb_ivb"].values
                fb_vaa_g = gr["fb_vaa"].values

                vsplit  = np.where(np.isfinite(fb_ivb_g),
                                   fb_ivb_g - gr["ivb"].values, GYRO_VSPLIT_MEAN)
                g_s    += GYRO_WEIGHTS["vsplit"] * _exp(
                    (vsplit - GYRO_VSPLIT_MEAN) / GYRO_VSPLIT_STD)

                vaadiff = np.where(np.isfinite(fb_vaa_g),
                                   np.abs(gr["adj_vaa"].values - fb_vaa_g), GYRO_VAADIFF_MEAN)
                g_s    += GYRO_WEIGHTS["vaadiff"] * _exp(
                    (vaadiff - GYRO_VAADIFF_MEAN) / GYRO_VAADIFF_STD)

                if "spin_efficiency" in gr.columns:
                    eff  = pd.to_numeric(gr["spin_efficiency"], errors="coerce")\
                             .fillna(GYRO_SPINEFF_MEAN).values
                    g_s += GYRO_WEIGHTS["spineff"] * _exp(
                        (GYRO_SPINEFF_MEAN - eff) / GYRO_SPINEFF_STD)

                s[gi] = np.maximum(s[gi], g_s)

        scores[idx] = s

    return scores


# ─── HIGH-iVB SWEEPER TAGGING ─────────────────────────────────────────────────

def tag_high_ivb_sweepers(df: pd.DataFrame, arm_models: dict = None) -> pd.DataFrame:
    df = df.copy()
    sw_mask = df["pitch_type"].isin({"ST", "SW"})
    if not sw_mask.any() or "ivb" not in df.columns:
        return df

    ivb = df.loc[sw_mask, "ivb"].values
    hb  = df.loc[sw_mask, "hb"].values

    if arm_models and "ST" in arm_models and "ivb" in arm_models["ST"] \
            and "arm_angle" in df.columns:
        aa           = df.loc[sw_mask, "arm_angle"].values
        expected_ivb = np.polyval(arm_models["ST"]["ivb"], aa)
        ivb_above    = ivb - expected_ivb
    else:
        ivb_above = ivb - (-4.0)

    hb_abs = np.abs(hb)
    hi_ivb = (ivb_above > HIGH_IVB_SW_IVB_THRESHOLD) & (hb_abs > HIGH_IVB_SW_HB_MIN)

    sw_idx = df.index[sw_mask]
    df.loc[sw_idx[hi_ivb], "pitch_type"] = "STH"
    return df


def untag_high_ivb_sweepers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["pitch_type"] == "STH", "pitch_type"] = "ST"
    return df


# ─── RANK NORMALIZATION (per pitch type) ─────────────────────────────────────

def fit_rank_norm(raw: np.ndarray, pitch_types: np.ndarray = None) -> dict:
    if pitch_types is None:
        return {"_global": np.sort(raw)}
    rn = {}
    for pt in np.unique(pitch_types):
        mask = pitch_types == pt
        rn[pt] = np.sort(raw[mask])
    return rn


def apply_rank_norm(raw: np.ndarray, rn: dict,
                    pitch_types: np.ndarray = None) -> np.ndarray:
    out = np.zeros(len(raw))
    if "_global" in rn:
        sv  = rn["_global"]
        pct = np.clip((np.searchsorted(sv, raw, side="left") + 0.5) / len(sv),
                      0.001, 0.999)
        return 100.0 + 10.0 * scipy_stats.norm.ppf(pct)
    for pt, sv in rn.items():
        if pitch_types is None:
            mask = np.ones(len(raw), dtype=bool)
        else:
            mask = pitch_types == pt
        if not mask.any():
            continue
        pct = np.clip(
            (np.searchsorted(sv, raw[mask], side="left") + 0.5) / len(sv),
            0.001, 0.999
        )
        out[mask] = 100.0 + 10.0 * scipy_stats.norm.ppf(pct)
    seen = set(rn.keys()) if pitch_types is not None else set()
    if pitch_types is not None:
        unseen = ~np.isin(pitch_types, list(seen))
        if unseen.any():
            all_sorted = np.sort(np.concatenate(list(rn.values())))
            pct = np.clip(
                (np.searchsorted(all_sorted, raw[unseen], side="left") + 0.5) / len(all_sorted),
                0.001, 0.999
            )
            out[unseen] = 100.0 + 10.0 * scipy_stats.norm.ppf(pct)
    return out


# ─── HARD VELOCITY FLOOR ──────────────────────────────────────────────────────

def apply_velo_floor(scores: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    scores  = scores.copy()
    fb_mask = df["pitch_type"].isin(FB_TYPES).values
    if fb_mask.any():
        v   = df.loc[fb_mask, "release_speed"].values
        cap = 100.0 + (v - FB_VELO_BASELINE) * FB_VELO_RATE
        scores[fb_mask] = np.minimum(scores[fb_mask], cap)
    return scores


# ─── FULL FEATURE PIPELINE ────────────────────────────────────────────────────

def add_physics_features(df, vaa_coeffs=None, league_stats=None,
                          arm_models=None, haa_stats=None):
    df = df.copy()

    # VAA / HAA (correct kinematic formula)
    vaa, haa = compute_vaa_haa(df)
    df["_vaa"] = vaa
    df["_haa"] = haa
    if vaa_coeffs is None:
        vaa_coeffs = fit_vaa_adjustment(df)
    df = apply_vaa_adjustment(df, vaa_coeffs)

    # Movement in inches
    df["ivb"] = df["pfx_z"] * 12.0
    df["hb"]  = df["pfx_x"] * 12.0

    # Fix HB sign: positive = arm side for both handedness
    df = fix_hb_handedness(df)

    # Spin axis as number
    if "spin_axis" in df.columns:
        df["spin_axis_num"] = pd.to_numeric(df["spin_axis"], errors="coerce")

    # Extension default if missing
    if "release_extension" not in df.columns:
        df["release_extension"] = 6.2

    # Arm angle (corrected for handedness)
    df["arm_angle"] = compute_arm_angle(df)

    if league_stats is None:
        league_stats = fit_league_stats(df)
    if arm_models is None:
        arm_models = fit_arm_angle_models(df)
    if haa_stats is None:
        haa_stats = fit_haa_stats(df)

    df = compute_fb_reference(df)

    return df, vaa_coeffs, league_stats, arm_models, haa_stats


# ─── TRAIN ────────────────────────────────────────────────────────────────────

def train(df, status_fn=None):
    df = df.copy()
    df["pitch_type"] = df["pitch_type"].fillna("").astype(str).str.strip()
    df = df[df["pitch_type"].str.len() > 0].reset_index(drop=True)

    if status_fn: status_fn("Computing physics features…", 0.10)
    df, vaa_c, lg, am, hs = add_physics_features(df)

    if status_fn: status_fn("Scoring all pitches…", 0.40)
    raw = compute_stuff_raw(df, lg, arm_models=am, haa_stats=hs)

    if status_fn: status_fn("Fitting rank normalization…", 0.70)
    df = tag_high_ivb_sweepers(df, arm_models=am)
    rn = fit_rank_norm(raw, df["pitch_type"].values)

    if status_fn: status_fn("Saving model…", 0.90)
    with open(MODEL_PATH, "wb") as f: pickle.dump({"formula": True}, f)
    with open(NORM_PATH,  "wb") as f: pickle.dump(rn, f)
    with open(VAA_PATH,   "wb") as f: pickle.dump(vaa_c, f)
    with open(AUX_PATH,   "wb") as f: pickle.dump(
        {"league_stats": lg, "arm_models": am, "haa_stats": hs}, f)
    (MODEL_DIR / "physics_version.txt").write_text(PHYSICS_VERSION)

    test  = apply_rank_norm(raw, rn, df["pitch_type"].values)
    test  = apply_velo_floor(test, df)
    df    = untag_high_ivb_sweepers(df)
    if status_fn: status_fn("✓ Done.", 1.0)
    return {
        "n_pitches":  len(df),
        "model_type": f"Direct Physics Formula {PHYSICS_VERSION}",
        "score_p99":  round(float(np.percentile(test, 99)), 1),
        "score_p95":  round(float(np.percentile(test, 95)), 1),
        "score_p50":  round(float(np.percentile(test, 50)), 1),
        "score_p05":  round(float(np.percentile(test, 5)),  1),
        "score_p01":  round(float(np.percentile(test, 1)),  1),
    }


# ─── LOAD / SCORE ─────────────────────────────────────────────────────────────

def model_is_trained():
    return NORM_PATH.exists() and VAA_PATH.exists()


def load_model():
    for p in [NORM_PATH, VAA_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run training first.")
    try:
        with open(NORM_PATH, "rb") as f: rn = pickle.load(f)
        with open(VAA_PATH,  "rb") as f: vc = pickle.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Model files are incompatible with the current code version "
            f"({PHYSICS_VERSION}). Please retrain. Error: {e}"
        )
    lg = am = hs = None
    if AUX_PATH.exists():
        try:
            with open(AUX_PATH, "rb") as f:
                aux = pickle.load(f)
                lg  = aux.get("league_stats")
                am  = aux.get("arm_models")
                hs  = aux.get("haa_stats")
        except Exception:
            pass
    return rn, vc, lg, am, hs


_DROP_BEFORE_SCORE = [
    "arm_angle", "spin_axis_num",
    "fb_ivb", "fb_hb", "fb_velo", "fb_vaa",
    "_vaa", "_haa",
    "adj_vaa",
    "ivb", "hb",
]


def score(df, rank_norm=None, vaa_coeffs=None, league_stats=None,
          arm_models=None, haa_stats=None, **_):
    if rank_norm is None:
        rank_norm, vaa_coeffs, league_stats, arm_models, haa_stats = load_model()

    df = df.copy()
    df["pitch_type"] = df["pitch_type"].fillna("").astype(str).str.strip()
    df = df[df["pitch_type"].str.len() > 0].reset_index(drop=True)

    df = df.drop(columns=[c for c in _DROP_BEFORE_SCORE if c in df.columns],
                 errors="ignore")
    df, vaa_coeffs, league_stats, am, hs = add_physics_features(
        df, vaa_coeffs=vaa_coeffs, league_stats=league_stats,
        arm_models=arm_models, haa_stats=haa_stats)

    raw    = compute_stuff_raw(df, league_stats, arm_models=am, haa_stats=hs)
    df     = tag_high_ivb_sweepers(df, arm_models=am)
    s      = apply_rank_norm(raw, rank_norm, df["pitch_type"].values)
    s      = apply_velo_floor(s, df)
    df     = untag_high_ivb_sweepers(df)

    df = df.copy()
    df["stuff_plus"] = np.round(s, 1)
    return df


def score_from_csv(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.strip().str.lower()
    req = ["release_speed","pfx_x","pfx_z","release_pos_x","release_pos_z",
           "vx0","vy0","vz0","ax","ay","az","plate_x","plate_z",
           "release_spin_rate","spin_axis","spin_efficiency","delta_run_exp"]
    for c in req:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "pitch_type" not in df.columns:
        raise ValueError("CSV must have a 'pitch_type' column")
    df["pitch_type"] = df["pitch_type"].fillna("").str.upper().str.strip()
    df = df[df["pitch_type"].str.len() > 0]
    df = df.dropna(subset=[c for c in req if c in df.columns], how="any")
    return score(df)
