"""
metrics.py — Bridge between model_trainer (physics Stuff+) and the card UI.

model_trainer.py owns all scoring logic:
  - Physics formula with domain-knowledge weights per pitch type
  - Arm-angle adjusted iVB/HB/VAA (VAA via correct Statcast kinematic formula)
  - HB sign: positive = arm side for both handedness
  - Gyro slider detection + separate scoring path
  - High-iVB sweeper detection
  - Offspeed graded on FB tunnel (depth gap, velo gap, HB fade)
  - Sinker: arm-side run + depth + outlier bonus
  - Rank normalization → N(100, 10)
  - Hard velocity floor for slow fastballs

This module just wraps train/score and builds the card summary dict.
"""

import numpy as np
import pandas as pd
import model_trainer

# Re-export for display
PITCH_NAMES = {
    'FF': 'Four-Seam', 'SI': 'Sinker',    'FC': 'Cutter',
    'SL': 'Slider',    'CU': 'Curveball', 'CH': 'Changeup',
    'FS': 'Splitter',  'KC': 'Knuckle Curve', 'ST': 'Sweeper',
    'SV': 'Slurve',    'KN': 'Knuckleball',   'CS': 'Slow Curve',
    'EP': 'Eephus',    'SC': 'Screwball',      'FO': 'Forkball',
    'SW': 'Sweeper',
}


def model_is_trained():
    return model_trainer.model_is_trained()


def train_model(df, status_fn=None):
    return model_trainer.train(df, status_fn=status_fn)


def train_model_streaming(chunks, status_fn=None):
    """Train on a list of chunk DataFrames without loading the full season into RAM."""
    return model_trainer.train_streaming(chunks, status_fn=status_fn)


def score_pitches(df):
    return model_trainer.score(df)


def score_from_csv(df):
    return model_trainer.score_from_csv(df)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _spin_axis_to_clock(degrees: float) -> str:
    """
    Convert spin axis in degrees to clock-face notation.
    Per Statcast: 180° = 12:00, each hour = 30°.
    Formula: Clock Hour = (spin_axis / 30 + 6) mod 12
    Returns string like '12:00', '1:30', '6:00' etc.
    """
    if degrees is None or (isinstance(degrees, float) and np.isnan(degrees)):
        return chr(8212)
    # Total clock-minutes from 12:00
    total_hours    = (degrees / 30.0 + 6.0) % 12.0
    hour_part      = int(total_hours)
    minute_part    = int(round((total_hours - hour_part) * 60))
    # Handle 60-minute rollover
    if minute_part == 60:
        hour_part  += 1
        minute_part = 0
    if hour_part == 0:
        hour_part   = 12
    return f"{hour_part}:{minute_part:02d}"


# ═════════════════════════════════════════════════════════════════════════════
# PITCHER CARD SUMMARY BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_pitcher_summary(df, pitcher_id=None):
    """Aggregate pitch-level scored data into the full player card structure."""
    if pitcher_id is not None and 'pitcher' in df.columns:
        pdf = df[df['pitcher'] == pitcher_id].copy()
    else:
        pdf = df.copy()
    if len(pdf) == 0:
        raise ValueError("No data for this pitcher.")

    # stuff_plus is the canonical column name from model_trainer
    stuff_col = 'stuff_plus' if 'stuff_plus' in pdf.columns else 'tjstuff_plus'

    pitcher_name = pdf['player_name'].iloc[0] if 'player_name' in pdf.columns else 'Unknown'
    pitcher_hand = pdf['p_throws'].iloc[0] if 'p_throws' in pdf.columns else 'R'
    total = len(pdf)

    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip',
                    'hit_into_play', 'foul_bunt', 'missed_bunt', 'bunt_foul_tip']
    whiff_events = ['swinging_strike', 'swinging_strike_blocked']

    def safe_pct(n, d):
        return round(n / d * 100, 1) if d > 0 else 0.0

    def in_zone(s):
        if 'zone' in s.columns:
            return s['zone'].isin(range(1, 10)).sum()
        return ((s['plate_x'].abs() < 0.83) & s['plate_z'].between(1.5, 3.5)).sum()

    def outside_zone(s):
        if 'zone' in s.columns:
            return s[~s['zone'].isin(range(1, 10))]
        return s[~((s['plate_x'].abs() < 0.83) & s['plate_z'].between(1.5, 3.5))]

    # iVB in inches (pfx_z * 12)
    if 'ivb' not in pdf.columns and 'pfx_z' in pdf.columns:
        pdf['ivb'] = pdf['pfx_z'] * 12

    # HB: arm-side-positive for both handedness.
    # model_trainer.score() already calls fix_hb_handedness(), so the 'hb'
    # column in a scored df is already corrected. If we're working with raw
    # pfx_x (e.g. pre-scoring path), compute and fix it here.
    if 'hb' not in pdf.columns and 'pfx_x' in pdf.columns:
        pdf['hb'] = pdf['pfx_x'] * 12
        pdf = model_trainer.fix_hb_handedness(pdf)

    # VAA / HAA using the correct kinematic formula
    if '_vaa' not in pdf.columns and 'vx0' in pdf.columns:
        vaa, haa = model_trainer.compute_vaa_haa(pdf)
        pdf['_vaa'] = vaa
        pdf['_haa'] = haa
    if 'adj_vaa' not in pdf.columns and '_vaa' in pdf.columns:
        pdf['adj_vaa'] = pdf['_vaa']   # fallback (no slot adjustment)

    # Spin axis for display
    if 'spin_axis_num' not in pdf.columns:
        if 'spin_axis' in pdf.columns:
            pdf['spin_axis_num'] = pd.to_numeric(pdf['spin_axis'], errors='coerce')
        elif 'ivb' in pdf.columns and 'hb' in pdf.columns:
            pdf['spin_axis_num'] = np.degrees(np.arctan2(pdf['hb'], pdf['ivb'])) % 360

    # Arm angle — uses abs(release_pos_x), no handedness flip needed
    if 'arm_angle' not in pdf.columns and 'release_pos_x' in pdf.columns:
        pdf['arm_angle'] = model_trainer.compute_arm_angle(pdf)

    # ── Header table ─────────────────────────────────────────────────────────
    header_rows = []
    for pt in pdf['pitch_type'].unique():
        if pd.isna(pt) or str(pt).strip() == '':
            continue
        sub = pdf[pdf['pitch_type'] == pt]
        n = len(sub)

        has_spin = 'release_spin_rate' in sub.columns and sub['release_spin_rate'].notna().any()
        has_vaa  = '_vaa' in sub.columns and sub['_vaa'].notna().any()
        has_haa  = '_haa' in sub.columns and sub['_haa'].notna().any()
        has_axis = 'spin_axis_num' in sub.columns and sub['spin_axis_num'].notna().any()

        stuff_val = (int(round(sub[stuff_col].mean()))
                     if stuff_col in sub.columns and sub[stuff_col].notna().any()
                     else chr(8212))

        header_rows.append({
            'Pitch':    PITCH_NAMES.get(pt, pt),
            'pt_code':  pt,
            'Count':    n,
            'Pitch%':   round(n / total * 100, 1),
            'Velocity': round(sub['release_speed'].mean(), 1),
            'iVB':      round(sub['ivb'].mean(), 1) if 'ivb' in sub.columns else chr(8212),
            'HB':       round(sub['hb'].mean(), 1)  if 'hb'  in sub.columns else chr(8212),
            'Spin':     int(round(sub['release_spin_rate'].mean())) if has_spin else chr(8212),
            'VAA':      round(sub['_vaa'].mean(), 1) if has_vaa else chr(8212),
            'HAA':      round(sub['_haa'].mean(), 1) if has_haa else chr(8212),
            'vRel':     round(sub['release_pos_z'].mean(), 1),
            'hRel':     round(sub['release_pos_x'].mean(), 1),
            'Ext.':     round(sub['release_extension'].mean(), 1)
                        if 'release_extension' in sub.columns else chr(8212),
            'Axis':     _spin_axis_to_clock(sub['spin_axis_num'].mean()) if has_axis else chr(8212),
            # Column keyed as 'Stuff+' for render_header_table
            'Stuff+':   stuff_val,
        })

    header_df = (pd.DataFrame(header_rows)
                   .sort_values('Count', ascending=False)
                   .reset_index(drop=True))

    # ── Usage LHH / RHH ──────────────────────────────────────────────────────
    usage_lhh, usage_rhh = {}, {}
    if 'stand' in pdf.columns:
        for side, ud in [('L', usage_lhh), ('R', usage_rhh)]:
            sdf = pdf[pdf['stand'] == side]
            st_ = len(sdf)
            for pt in sdf['pitch_type'].unique():
                if pd.isna(pt):
                    continue
                ud[pt] = round(len(sdf[sdf['pitch_type'] == pt]) / st_ * 100, 1) if st_ > 0 else 0

    # ── Metrics table ─────────────────────────────────────────────────────────
    metrics_rows = []
    for pt in header_df['pt_code'].values:
        sub = pdf[pdf['pitch_type'] == pt]
        n = len(sub)
        z_pct = safe_pct(in_zone(sub), n)
        out   = outside_zone(sub)
        c_pct = 0.0
        if 'description' in sub.columns and len(out) > 0:
            c_pct = safe_pct(out[out['description'].isin(swing_events)].shape[0], len(out))
        w_pct = 0.0
        if 'description' in sub.columns:
            w_pct = safe_pct(
                sub['description'].isin(whiff_events).sum(),
                sub['description'].isin(swing_events).sum())
        xw = chr(8212)
        if 'estimated_woba_using_speedangle' in sub.columns:
            ct = sub[sub['estimated_woba_using_speedangle'].notna()]
            if len(ct) > 0:
                xw = round(ct['estimated_woba_using_speedangle'].mean(), 3)
        metrics_rows.append({
            'Pitch':   PITCH_NAMES.get(pt, pt),
            'pt_code': pt,
            'Zone%':   z_pct,
            'Chase%':  c_pct,
            'Whiff%':  w_pct,
            'xwOBA':   xw,
        })
    metrics_df = pd.DataFrame(metrics_rows)

    # ── Overall stats ─────────────────────────────────────────────────────────
    overall_zone  = safe_pct(in_zone(pdf), total)
    overall_chase = 0.0
    overall_whiff = 0.0
    if 'description' in pdf.columns:
        oa = outside_zone(pdf)
        if len(oa) > 0:
            overall_chase = safe_pct(
                oa[oa['description'].isin(swing_events)].shape[0], len(oa))
        overall_whiff = safe_pct(
            pdf['description'].isin(whiff_events).sum(),
            pdf['description'].isin(swing_events).sum())
    overall_xwoba = chr(8212)
    if 'estimated_woba_using_speedangle' in pdf.columns:
        c = pdf[pdf['estimated_woba_using_speedangle'].notna()]
        if len(c) > 0:
            overall_xwoba = round(c['estimated_woba_using_speedangle'].mean(), 3)

    overall_stuff = (int(round(pdf[stuff_col].mean()))
                     if stuff_col in pdf.columns and pdf[stuff_col].notna().any()
                     else 100)

    # ── Movement data for plot ────────────────────────────────────────────────
    # hb is already arm-side-positive from the scoring pipeline
    mvmt_cols = ['pitch_type']
    if 'ivb'       in pdf.columns: mvmt_cols.append('ivb')
    if 'hb'        in pdf.columns: mvmt_cols.append('hb')
    if 'arm_angle' in pdf.columns: mvmt_cols.append('arm_angle')
    mvmt = pdf[mvmt_cols].copy().dropna(subset=['ivb', 'hb'])

    # Per-pitch-type arm angles — used for arm-angle-cheating penalty in scoring
    # and for per-line display on the movement plot
    arm_angles_by_pitch = {}
    if 'arm_angle' in pdf.columns:
        for pt, grp in pdf.groupby('pitch_type'):
            aa = grp['arm_angle'].dropna().mean()
            if np.isfinite(aa):
                arm_angles_by_pitch[pt] = round(aa, 1)

    return {
        'header_table':       header_df,
        'usage_lhh':          usage_lhh,
        'usage_rhh':          usage_rhh,
        'metrics_table':      metrics_df,
        'movement_data':      mvmt,
        'arm_angle':          round(pdf['arm_angle'].mean(), 1)
                              if 'arm_angle' in pdf.columns else 45.0,
        'arm_angles_by_pitch': arm_angles_by_pitch,
        'pitcher_name':       pitcher_name,
        'pitcher_hand':       pitcher_hand,
        'total_pitches':      total,
        'overall_zone':       overall_zone,
        'overall_chase':      overall_chase,
        'overall_whiff':      overall_whiff,
        'overall_xwoba':      overall_xwoba,
        'overall_stuff':      overall_stuff,
    }
