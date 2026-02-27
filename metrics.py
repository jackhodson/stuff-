"""
metrics.py — Bridge between model_trainer (physics Stuff+) and the card UI.

model_trainer.py owns all scoring logic:
  - Physics formula with domain-knowledge weights per pitch type
  - Arm-angle adjusted iVB/HB/VAA
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
    'FF':'Four-Seam','SI':'Sinker','FC':'Cutter','SL':'Slider',
    'CU':'Curveball','CH':'Changeup','FS':'Splitter','KC':'Knuckle Curve',
    'ST':'Sweeper','SV':'Slurve','KN':'Knuckleball','CS':'Slow Curve',
    'EP':'Eephus','SC':'Screwball','FO':'Forkball','SW':'Sweeper',
}


def model_is_trained():
    return model_trainer.model_is_trained()


def train_model(df, status_fn=None):
    """Train the physics-formula Stuff+ model on a full season."""
    return model_trainer.train(df, status_fn=status_fn)


def score_pitches(df):
    """Score a DataFrame of Statcast pitches using the pre-trained model."""
    return model_trainer.score(df)


def score_from_csv(df):
    """Score raw CSV — handles column cleaning."""
    return model_trainer.score_from_csv(df)


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

    # Use stuff_plus from model_trainer (not tjstuff_plus)
    stuff_col = 'stuff_plus' if 'stuff_plus' in pdf.columns else 'tjstuff_plus'

    pitcher_name = pdf['player_name'].iloc[0] if 'player_name' in pdf.columns else 'Unknown'
    pitcher_hand = pdf['p_throws'].iloc[0] if 'p_throws' in pdf.columns else 'R'
    total = len(pdf)

    swing_events = ['swinging_strike','swinging_strike_blocked','foul','foul_tip',
                    'hit_into_play','foul_bunt','missed_bunt','bunt_foul_tip']
    whiff_events = ['swinging_strike','swinging_strike_blocked']

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

    # Derive ivb/hb for display if not already present
    if 'ivb' not in pdf.columns and 'pfx_z' in pdf.columns:
        pdf['ivb'] = pdf['pfx_z'] * 12
    if 'hb' not in pdf.columns and 'pfx_x' in pdf.columns:
        pdf['hb'] = pdf['pfx_x'] * 12

    # VAA/HAA for display
    if '_vaa' not in pdf.columns and 'vx0' in pdf.columns:
        vaa, haa = model_trainer.compute_vaa_haa(pdf)
        pdf['_vaa'] = vaa
        pdf['_haa'] = haa
    if 'adj_vaa' not in pdf.columns and '_vaa' in pdf.columns:
        pdf['adj_vaa'] = pdf['_vaa']  # raw VAA for display fallback

    # Spin axis for display
    if 'spin_axis_num' not in pdf.columns:
        if 'spin_axis' in pdf.columns:
            pdf['spin_axis_num'] = pd.to_numeric(pdf['spin_axis'], errors='coerce')
        elif 'ivb' in pdf.columns and 'hb' in pdf.columns:
            pdf['spin_axis_num'] = np.degrees(np.arctan2(pdf['hb'], pdf['ivb'])) % 360

    # Arm angle for display
    if 'arm_angle' not in pdf.columns:
        pdf['arm_angle'] = np.degrees(
            np.arctan2(pdf['release_pos_z'].values - 5.5,
                       np.abs(pdf['release_pos_x'].values)))

    # Header table
    header_rows = []
    for pt in pdf['pitch_type'].unique():
        if pd.isna(pt) or str(pt).strip() == '':
            continue
        sub = pdf[pdf['pitch_type'] == pt]
        n = len(sub)
        has_spin = 'release_spin_rate' in sub.columns and sub['release_spin_rate'].notna().any()
        has_vaa = '_vaa' in sub.columns and sub['_vaa'].notna().any()
        has_haa = '_haa' in sub.columns and sub['_haa'].notna().any()
        has_axis = 'spin_axis_num' in sub.columns and sub['spin_axis_num'].notna().any()

        stuff_val = int(round(sub[stuff_col].mean())) if stuff_col in sub.columns and sub[stuff_col].notna().any() else chr(8212)

        header_rows.append({
            'Pitch': PITCH_NAMES.get(pt, pt),
            'pt_code': pt,
            'Count': n,
            'Pitch%': round(n / total * 100, 1),
            'Velocity': round(sub['release_speed'].mean(), 1),
            'iVB': round(sub['ivb'].mean(), 1) if 'ivb' in sub.columns else chr(8212),
            'HB': round(sub['hb'].mean(), 1) if 'hb' in sub.columns else chr(8212),
            'Spin': int(round(sub['release_spin_rate'].mean())) if has_spin else chr(8212),
            'VAA': round(sub['_vaa'].mean(), 1) if has_vaa else chr(8212),
            'HAA': round(sub['_haa'].mean(), 1) if has_haa else chr(8212),
            'vRel': round(sub['release_pos_z'].mean(), 1),
            'hRel': round(sub['release_pos_x'].mean(), 1),
            'Ext.': round(sub['release_extension'].mean(), 1) if 'release_extension' in sub.columns else chr(8212),
            'Axis': int(round(sub['spin_axis_num'].mean())) if has_axis else chr(8212),
            'tjStuff+': stuff_val,
        })
    header_df = pd.DataFrame(header_rows).sort_values('Count', ascending=False).reset_index(drop=True)

    # Usage LHH / RHH
    usage_lhh, usage_rhh = {}, {}
    if 'stand' in pdf.columns:
        for side, ud in [('L', usage_lhh), ('R', usage_rhh)]:
            sdf = pdf[pdf['stand'] == side]
            st = len(sdf)
            for pt in sdf['pitch_type'].unique():
                if pd.isna(pt):
                    continue
                ud[pt] = round(len(sdf[sdf['pitch_type'] == pt]) / st * 100, 1) if st > 0 else 0

    # Metrics table
    metrics_rows = []
    for pt in header_df['pt_code'].values:
        sub = pdf[pdf['pitch_type'] == pt]
        n = len(sub)
        z_pct = safe_pct(in_zone(sub), n)
        out = outside_zone(sub)
        c_pct = 0.0
        if 'description' in sub.columns and len(out) > 0:
            c_pct = safe_pct(out[out['description'].isin(swing_events)].shape[0], len(out))
        w_pct = 0.0
        if 'description' in sub.columns:
            w_pct = safe_pct(sub['description'].isin(whiff_events).sum(),
                             sub['description'].isin(swing_events).sum())
        xw = chr(8212)
        if 'estimated_woba_using_speedangle' in sub.columns:
            ct = sub[sub['estimated_woba_using_speedangle'].notna()]
            if len(ct) > 0:
                xw = round(ct['estimated_woba_using_speedangle'].mean(), 3)
        metrics_rows.append({
            'Pitch': PITCH_NAMES.get(pt, pt), 'pt_code': pt,
            'Zone%': z_pct, 'Chase%': c_pct, 'Whiff%': w_pct, 'xwOBA': xw,
        })
    metrics_df = pd.DataFrame(metrics_rows)

    # Overall stats
    overall_zone = safe_pct(in_zone(pdf), total)
    overall_chase = 0.0
    overall_whiff = 0.0
    if 'description' in pdf.columns:
        oa = outside_zone(pdf)
        if len(oa) > 0:
            overall_chase = safe_pct(oa[oa['description'].isin(swing_events)].shape[0], len(oa))
        overall_whiff = safe_pct(pdf['description'].isin(whiff_events).sum(),
                                 pdf['description'].isin(swing_events).sum())
    overall_xwoba = chr(8212)
    if 'estimated_woba_using_speedangle' in pdf.columns:
        c = pdf[pdf['estimated_woba_using_speedangle'].notna()]
        if len(c) > 0:
            overall_xwoba = round(c['estimated_woba_using_speedangle'].mean(), 3)

    overall_stuff = int(round(pdf[stuff_col].mean())) if stuff_col in pdf.columns and pdf[stuff_col].notna().any() else 100

    # Movement data for plot
    mvmt = pdf[['pitch_type']].copy()
    if 'ivb' in pdf.columns:
        mvmt['ivb'] = pdf['ivb']
    if 'hb' in pdf.columns:
        mvmt['hb'] = pdf['hb']
    mvmt = mvmt.dropna()

    return {
        'header_table': header_df,
        'usage_lhh': usage_lhh,
        'usage_rhh': usage_rhh,
        'metrics_table': metrics_df,
        'movement_data': mvmt,
        'arm_angle': round(pdf['arm_angle'].mean(), 1) if 'arm_angle' in pdf.columns else 45.0,
        'pitcher_name': pitcher_name,
        'pitcher_hand': pitcher_hand,
        'total_pitches': total,
        'overall_zone': overall_zone,
        'overall_chase': overall_chase,
        'overall_whiff': overall_whiff,
        'overall_xwoba': overall_xwoba,
        'overall_stuff': overall_stuff,
    }
