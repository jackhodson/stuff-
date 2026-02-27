"""
app.py - Pitcher Player Card Dashboard

Uses model_trainer.py for physics-formula Stuff+ scoring.
Model ships pre-trained (pickle files in data/model/).
Supports live Savant API lookup and CSV upload.

FIXES:
  - All "tjStuff+" labels → "Stuff+"
  - Movement plot: HB axis labeled arm side / glove side (handedness-aware)
  - Movement plot: arm angle derived correctly from release position
  - Pitcher search dropdown for autocomplete
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from datetime import datetime

from data_fetcher import (
    fetch_pitcher_season, fetch_pitcher_live, search_pitcher,
    season_chunk_schedule, fetch_chunk, clear_chunk_cache,
)
from metrics import (
    model_is_trained, train_model, train_model_streaming, score_pitches,
    score_from_csv, build_pitcher_summary, PITCH_NAMES,
)
import model_trainer

st.set_page_config(page_title="Pitcher Card", page_icon="⚾", layout="wide",
                   initial_sidebar_state="expanded")

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700;800;900&display=swap');
:root{--bg:#06090f;--bg-card:#0d1117;--bg-srf:#161b22;--bdr:#21262d;--txt:#e6edf3;--dim:#7d8590;--acc:#58a6ff;--acc2:#bc8cff;--red:#ff7b72;--org:#ffa657;--grn:#56d364;--yel:#e3b341;--cyn:#79c0ff;--pnk:#f778ba}
.stApp{background:var(--bg);color:var(--txt)}
.main .block-container{padding-top:.5rem;max-width:1440px}
[data-testid="stSidebar"]{background:var(--bg-card);border-right:1px solid var(--bdr)}
h1,h2,h3,h4{font-family:'DM Sans',sans-serif!important}
.hero{background:linear-gradient(135deg,#0d1117,#161b22,#0d1117);border:1px solid #30363d;border-radius:16px;padding:1.75rem 2rem;margin-bottom:1.25rem;display:flex;align-items:center;justify-content:space-between;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#58a6ff,#bc8cff,transparent)}
.hero-name{font-family:'DM Sans';font-weight:900;font-size:2.5rem;color:#e6edf3;letter-spacing:-.03em;line-height:1}
.hero-sub{font-family:'JetBrains Mono';font-size:.82rem;color:var(--dim);margin-top:4px}
.stuff-badge{background:linear-gradient(135deg,#58a6ff,#bc8cff);color:#fff;border-radius:14px;padding:.6rem 1.6rem;text-align:center;font-family:'JetBrains Mono'}
.stuff-val{font-weight:700;font-size:2.4rem;line-height:1}
.stuff-label{font-size:.6rem;font-weight:500;text-transform:uppercase;letter-spacing:.12em;opacity:.85}
.pills{display:flex;gap:.6rem;margin-bottom:1.25rem;flex-wrap:wrap}
.pill{background:var(--bg-card);border:1px solid var(--bdr);border-radius:10px;padding:.65rem 1rem;text-align:center;flex:1;min-width:90px}
.pill-val{font-family:'JetBrains Mono';font-weight:700;font-size:1.3rem;color:var(--txt)}
.pill-lbl{font-family:'DM Sans';font-weight:600;font-size:.62rem;color:var(--dim);text-transform:uppercase;letter-spacing:.08em;margin-top:2px}
.card{background:var(--bg-card);border:1px solid var(--bdr);border-radius:14px;padding:1.25rem;margin-bottom:1rem}
.card-hdr{font-family:'DM Sans';font-weight:800;font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;color:var(--dim);margin-bottom:.75rem;display:flex;align-items:center;gap:8px}
.card-hdr .dot{width:6px;height:6px;border-radius:50%;background:var(--acc)}
.ptable{width:100%;border-collapse:separate;border-spacing:0;font-family:'JetBrains Mono';font-size:.76rem}
.ptable thead th{background:var(--bg-srf);color:var(--dim);font-weight:600;font-size:.66rem;text-transform:uppercase;letter-spacing:.05em;padding:.55rem .45rem;text-align:center;border-bottom:2px solid var(--acc);white-space:nowrap;position:sticky;top:0}
.ptable thead th:first-child{text-align:left;border-radius:8px 0 0 0}
.ptable thead th:last-child{border-radius:0 8px 0 0}
.ptable tbody td{padding:.5rem .45rem;text-align:center;border-bottom:1px solid #21262d;color:var(--txt)}
.ptable tbody td:first-child{text-align:left;font-weight:600}
.ptable tbody tr:hover{background:rgba(88,166,255,.04)}
.stuff-elite{color:#56d364;font-weight:700}.stuff-plus{color:#7ee787;font-weight:600}.stuff-avg{color:#7d8590}.stuff-minus{color:#e3b341;font-weight:600}.stuff-bad{color:#ff7b72;font-weight:700}
.pdot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
#MainMenu,footer,header{visibility:hidden}.stDeployButton{display:none}
</style>
""", unsafe_allow_html=True)

PITCH_COLORS = {
    'FF':'#ff7b72','SI':'#ffa657','FC':'#bc8cff','SL':'#e3b341','ST':'#d2a8ff',
    'SV':'#db8e00','CU':'#79c0ff','KC':'#58a6ff','CS':'#a5d6ff','CH':'#56d364',
    'FS':'#3fb950','KN':'#7d8590','EP':'#f778ba','SC':'#ff9bce','FO':'#39d353',
    'SW':'#d2a8ff',
}

def stuff_css(val):
    try: v = int(val)
    except: return 'stuff-avg'
    if v >= 115: return 'stuff-elite'
    if v >= 105: return 'stuff-plus'
    if v >= 95: return 'stuff-avg'
    if v >= 85: return 'stuff-minus'
    return 'stuff-bad'


# ── Charts ────────────────────────────────────────────────────────────────────

def make_movement_plot(mvmt, arm_angle, hand):
    """
    Movement plot with one arm-angle line per pitch type, color-coded to match
    the pitch scatter dots. Lines fan out from origin showing each pitch type's
    average release slot — gaps between lines expose arm-angle cheating.

    arm_angle: overall mean (used as fallback if per-pitch data missing)
    """
    fig = go.Figure()

    # ── Reference circles ────────────────────────────────────────────────────
    for r in [5, 10, 15, 20, 25]:
        t = np.linspace(0, 2 * np.pi, 120)
        fig.add_trace(go.Scatter(
            x=r * np.cos(t), y=r * np.sin(t), mode='lines',
            line=dict(color='rgba(33,38,45,0.7)', width=1),
            showlegend=False, hoverinfo='skip'))

    fig.add_hline(y=0, line_color='rgba(48,54,61,0.6)', line_width=1)
    fig.add_vline(x=0, line_color='rgba(48,54,61,0.6)', line_width=1)

    # ── Clock labels ─────────────────────────────────────────────────────────
    for deg, lbl in {0: '12', 90: '3', 180: '6', 270: '9'}.items():
        rad = math.radians(90 - deg)
        fig.add_annotation(
            x=27 * math.cos(rad), y=27 * math.sin(rad), text=lbl,
            font=dict(size=11, color='#484f58', family='JetBrains Mono'),
            showarrow=False)

    # ── Per-pitch-type arm angle lines ────────────────────────────────────────
    # Compute mean arm_angle per pitch type from the per-pitch column if available.
    # Line length scales slightly with pitch usage (more pitches = longer line),
    # making the dominant pitch type's line the most prominent.
    pitch_types_ordered = [pt for pt in mvmt['pitch_type'].unique()
                           if not pd.isna(pt)]
    has_per_pitch_aa = 'arm_angle' in mvmt.columns

    # Collect per-pitch-type angles for the annotation table
    pt_angles = {}
    for pt in pitch_types_ordered:
        sub = mvmt[mvmt['pitch_type'] == pt]
        if has_per_pitch_aa:
            aa = sub['arm_angle'].dropna().mean()
        else:
            aa = arm_angle  # fallback: single overall angle
        if np.isfinite(aa):
            pt_angles[pt] = round(aa, 1)

    # Draw lines — longer for dominant pitch types
    total_pitches = len(mvmt)
    for pt, aa in pt_angles.items():
        sub   = mvmt[mvmt['pitch_type'] == pt]
        frac  = len(sub) / max(total_pitches, 1)
        # Line length: 16 (rare pitch) to 23 (dominant pitch)
        length = 16 + 7 * frac
        rad    = math.radians(aa)
        lx     = length * math.cos(rad)
        ly     = length * math.sin(rad)
        color  = PITCH_COLORS.get(pt, '#7d8590')
        nm     = PITCH_NAMES.get(pt, pt)

        fig.add_trace(go.Scatter(
            x=[0, lx], y=[0, ly], mode='lines',
            name=f'{nm} arm',
            line=dict(color=color, width=2, dash='dot'),
            showlegend=False,
            hovertemplate=f'<b>{nm}</b><br>Arm Angle: {aa}°<extra></extra>'))

        # Angle label at tip of line
        fig.add_annotation(
            x=lx * 1.08, y=ly * 1.08,
            text=f'{aa}°',
            font=dict(size=8, color=color, family='JetBrains Mono'),
            showarrow=False, bgcolor='rgba(13,17,23,0.75)')

    # ── Pitch scatter ─────────────────────────────────────────────────────────
    for pt in pitch_types_ordered:
        s  = mvmt[mvmt['pitch_type'] == pt]
        c  = PITCH_COLORS.get(pt, '#7d8590')
        nm = PITCH_NAMES.get(pt, pt)
        fig.add_trace(go.Scatter(
            x=s['hb'], y=s['ivb'], mode='markers', name=nm,
            marker=dict(size=6, color=c, opacity=0.7,
                        line=dict(width=0.4, color='rgba(0,0,0,0.4)')),
            hovertemplate=f'<b>{nm}</b><br>HB: %{{x:.1f}}"<br>iVB: %{{y:.1f}}"<extra></extra>'))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(family='JetBrains Mono', color='#7d8590', size=10),
        title=dict(
            text='<b>Pitch Movement</b>',
            font=dict(size=14, color='#e6edf3', family='DM Sans'), x=0.01, y=0.98),
        xaxis=dict(
            title='← Glove Side  |  Arm Side →',
            range=[-28, 28], zeroline=False,
            gridcolor='rgba(33,38,45,0.3)', dtick=10),
        yaxis=dict(
            title='Induced Vert. Break (in)',
            range=[-28, 28], zeroline=False,
            gridcolor='rgba(33,38,45,0.3)', dtick=10,
            scaleanchor='x', scaleratio=1),
        legend=dict(
            orientation='h', y=-0.17, x=0.5, xanchor='center',
            font=dict(size=9), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=50, r=15, t=42, b=55),
        height=500, width=500)
    return fig


def make_usage_chart(usage_lhh, usage_rhh, header_df):
    pts = header_df['pt_code'].tolist()
    labels = [PITCH_NAMES.get(pt, pt) for pt in pts]
    lvals = [usage_lhh.get(pt, 0) for pt in pts]
    rvals = [usage_rhh.get(pt, 0) for pt in pts]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=[-v for v in lvals], orientation='h', name='vs LHH',
        marker_color=[PITCH_COLORS.get(pt, '#7d8590') for pt in pts], opacity=0.85,
        text=[f'{v}%' if v > 2 else '' for v in lvals], textposition='inside',
        textfont=dict(size=10, color='white', family='JetBrains Mono')))
    fig.add_trace(go.Bar(
        y=labels, x=rvals, orientation='h', name='vs RHH',
        marker_color=[PITCH_COLORS.get(pt, '#7d8590') for pt in pts], opacity=0.5,
        text=[f'{v}%' if v > 2 else '' for v in rvals], textposition='inside',
        textfont=dict(size=10, color='white', family='JetBrains Mono')))
    mx = max(max(lvals, default=0), max(rvals, default=0), 10) * 1.2
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(family='JetBrains Mono', color='#7d8590'),
        title=dict(
            text='<b>Pitch Usage — LHH vs RHH</b>',
            font=dict(size=14, color='#e6edf3', family='DM Sans'), x=0.01, y=0.98),
        barmode='overlay',
        xaxis=dict(
            range=[-mx, mx], zeroline=True, zerolinecolor='#30363d', zerolinewidth=2,
            tickvals=list(range(-60, 70, 20)),
            ticktext=[f'{abs(v)}%' for v in range(-60, 70, 20)],
            gridcolor='rgba(33,38,45,0.3)'),
        yaxis=dict(autorange='reversed'),
        legend=dict(
            orientation='h', y=-0.12, x=0.5, xanchor='center',
            font=dict(size=9), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=100, r=15, t=42, b=45),
        height=max(260, len(pts) * 48 + 90))
    return fig


# ── Table Renderers ───────────────────────────────────────────────────────────

def render_header_table(hdf):
    cols = ['Pitch', 'Count', 'Pitch%', 'Velocity', 'iVB', 'HB', 'Spin',
            'VAA', 'HAA', 'vRel', 'hRel', 'Ext.', 'Axis (Clock)', 'Stuff+']
    h = '<table class="ptable"><thead><tr>'
    for c in cols:
        h += f'<th>{c}</th>'
    h += '</tr></thead><tbody>'
    for _, r in hdf.iterrows():
        pt = r.get('pt_code', '')
        clr = PITCH_COLORS.get(pt, '#7d8590')
        h += '<tr>'
        for c in cols:
            # Map display column name to data column name
            data_key = 'tjStuff+' if 'tjStuff+' in r.index else 'Stuff+'
            if c == 'Axis (Clock)':
                v = r.get('Axis', chr(8212))
            elif c == 'Stuff+':
                v = r.get(data_key, chr(8212))
            else:
                v = r.get(c, chr(8212))
            if c == 'Pitch':
                h += f'<td><span class="pdot" style="background:{clr}"></span>{v}</td>'
            elif c == 'Stuff+':
                h += f'<td class="{stuff_css(v)}">{v}</td>'
            else:
                h += f'<td>{v}</td>'
        h += '</tr>'
    h += '</tbody></table>'
    return h


def render_metrics_table(mdf):
    cols = ['Pitch', 'Zone%', 'Chase%', 'Whiff%', 'xwOBA']
    h = '<table class="ptable"><thead><tr>'
    for c in cols:
        h += f'<th>{c}</th>'
    h += '</tr></thead><tbody>'
    for _, r in mdf.iterrows():
        pt = r.get('pt_code', '')
        clr = PITCH_COLORS.get(pt, '#7d8590')
        h += '<tr>'
        for c in cols:
            v = r.get(c, chr(8212))
            if c == 'Pitch':
                h += f'<td><span class="pdot" style="background:{clr}"></span>{v}</td>'
            else:
                h += f'<td>{v}</td>'
        h += '</tr>'
    h += '</tbody></table>'
    return h


# ── Card Renderer ─────────────────────────────────────────────────────────────

def render_card(summary, model_label):
    s = summary
    hand = "RHP" if s['pitcher_hand'] == 'R' else "LHP"
    st.markdown(f'''
    <div class="hero">
        <div><div class="hero-name">{s['pitcher_name']}</div>
        <div class="hero-sub">{hand} · {s['total_pitches']:,} pitches · {model_label}</div></div>
        <div class="stuff-badge">
            <div class="stuff-label">Overall Stuff+</div>
            <div class="stuff-val">{s['overall_stuff']}</div>
        </div>
    </div>''', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="pills">
        <div class="pill"><div class="pill-val">{s['overall_zone']}%</div><div class="pill-lbl">Zone%</div></div>
        <div class="pill"><div class="pill-val">{s['overall_chase']}%</div><div class="pill-lbl">Chase%</div></div>
        <div class="pill"><div class="pill-val">{s['overall_whiff']}%</div><div class="pill-lbl">Whiff%</div></div>
        <div class="pill"><div class="pill-val">{s['overall_xwoba']}</div><div class="pill-lbl">xwOBA Contact</div></div>
        <div class="pill"><div class="pill-val">{s['overall_stuff']}</div><div class="pill-lbl">Stuff+</div></div>
    </div>''', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-hdr"><span class="dot"></span>Pitch Arsenal</div>',
                unsafe_allow_html=True)
    st.markdown(render_header_table(s['header_table']), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(
            make_movement_plot(s['movement_data'], s['arm_angle'], s['pitcher_hand']),
            use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(
            make_usage_chart(s['usage_lhh'], s['usage_rhh'], s['header_table']),
            use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-hdr"><span class="dot"></span>Performance Metrics</div>',
                unsafe_allow_html=True)
    st.markdown(render_metrics_table(s['metrics_table']), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f'''
    <div style="text-align:center;margin-top:2rem;padding:1rem;font-family:'JetBrains Mono';font-size:.65rem;color:#484f58">
        Stuff+ · Physics Formula {model_trainer.PHYSICS_VERSION} · μ=100 σ=10 ·
        Arm-angle adjusted · Rank normalized · Data: Baseball Savant
    </div>''', unsafe_allow_html=True)


def render_landing():
    trained = model_is_trained()
    status = '✅ Model loaded (pre-trained)' if trained else '⚠️ No model — train or provide pickle files'
    st.markdown(f'''
    <div class="card" style="text-align:center;padding:3rem 2rem">
        <div style="font-size:3rem;margin-bottom:1rem">⚾</div>
        <div style="font-family:'DM Sans';font-weight:800;font-size:1.5rem;color:#e6edf3;margin-bottom:.75rem">
            Pitcher Player Card</div>
        <div style="font-family:'JetBrains Mono';font-size:.8rem;color:#7d8590;max-width:600px;margin:0 auto;line-height:1.7">
            <b style="color:#58a6ff">Model Status:</b> {status}<br><br>
            <b style="color:#56d364">Enter a pitcher name</b> and click <b>"Load Pitcher"</b>
            to pull data live from Savant and score with the physics Stuff+ model.<br><br>
            The model uses arm-angle adjusted movement, per-pitcher FB tunneling for
            offspeed, gyro slider detection, and rank normalization.
        </div>
    </div>''', unsafe_allow_html=True)


# ── Process + Score ───────────────────────────────────────────────────────────

def process_and_score(df, pitcher_id=None):
    scored = score_pitches(df)
    label = f"Physics {model_trainer.PHYSICS_VERSION}"
    summary = build_pitcher_summary(scored, pitcher_id)
    return summary, label


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('''
    <div style="font-family:'DM Sans';font-weight:900;font-size:1.2rem;color:#e6edf3;margin-bottom:.2rem">⚾ Pitcher Card</div>
    <div style="font-family:'JetBrains Mono';font-size:.68rem;color:#58a6ff;margin-bottom:1.25rem">Stuff+ Physics Engine</div>
    ''', unsafe_allow_html=True)
    st.markdown("---")
    data_mode = st.radio("Data Source", ["Live (Savant API)", "Upload CSV"], index=0)
    st.markdown("---")

    if model_is_trained():
        st.success("Model loaded ✅", icon="✅")
    else:
        st.warning("No model files found")

    uploaded_file = None
    train_btn = False
    load_btn = False
    pitcher_input = ""
    lookup_year = 2025
    days_back = None

    if data_mode == "Live (Savant API)":
        st.markdown("---")
        st.markdown('<div style="font-family:DM Sans;font-weight:700;font-size:.8rem;color:#e6edf3;margin-bottom:.4rem">Retrain Model</div>',
                    unsafe_allow_html=True)
        train_year = st.selectbox("Training Season", [2025, 2024, 2023], index=0)

        col_a, col_b = st.columns(2)
        with col_a:
            train_btn = st.button("Start / Resume", use_container_width=True, type="primary")
        with col_b:
            reset_btn = st.button("Reset", use_container_width=True)

        # Show download progress if a retrain is in progress
        if "train_schedule" in st.session_state:
            sched    = st.session_state.train_schedule
            n_done   = sum(1 for c in sched if c["done"])
            n_total  = len(sched)
            st.progress(n_done / n_total, text=f"{n_done}/{n_total} chunks downloaded")
            if n_done == n_total and not st.session_state.get("train_fit_done"):
                st.info("All chunks downloaded — fitting model…")
            elif n_done == n_total and st.session_state.get("train_fit_done"):
                st.success("Model trained ✅")

        st.markdown("---")
        st.markdown('<div style="font-family:DM Sans;font-weight:700;font-size:.8rem;color:#e6edf3;margin-bottom:.4rem">Pitcher Lookup</div>',
                    unsafe_allow_html=True)

        # ── Search with dropdown autocomplete ─────────────────────────────
        search_query = st.text_input(
            "Search Pitcher",
            placeholder="Start typing a name…",
            key="pitcher_search_input"
        )

        pitcher_input = ""
        selected_pitcher_id = None

        if search_query and len(search_query) >= 2:
            with st.spinner("Searching…"):
                search_results = search_pitcher(search_query)

            if len(search_results) > 0:
                # Build display options
                name_col = 'name' if 'name' in search_results.columns else \
                           ('player_name' if 'player_name' in search_results.columns else None)
                id_col   = 'id' if 'id' in search_results.columns else \
                           ('player_id' if 'player_id' in search_results.columns else None)

                if name_col and id_col:
                    options = search_results[[name_col, id_col]].dropna().head(10)
                    option_labels = options[name_col].tolist()

                    chosen_name = st.selectbox(
                        "Select Pitcher",
                        options=option_labels,
                        key="pitcher_dropdown"
                    )
                    if chosen_name:
                        row = options[options[name_col] == chosen_name].iloc[0]
                        pitcher_input    = chosen_name
                        selected_pitcher_id = int(row[id_col])
                else:
                    st.warning("Unexpected search result format.")
            else:
                st.info("No results found. Try Last, First format.")

        lookup_year = st.selectbox("Season", [2026, 2025, 2024], index=0)
        if lookup_year >= datetime.now().year:
            days_back = st.slider("Days Back", 7, 180, 45)
        load_btn = st.button("Load Pitcher", use_container_width=True, type="primary")
    else:
        uploaded_file = st.file_uploader("Upload Statcast CSV", type=['csv'])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ══════════════════════════════════════════════════════════════════════════════

if data_mode == "Live (Savant API)":

    # ── Reset button ─────────────────────────────────────────────────────────
    if reset_btn:
        if "train_schedule" in st.session_state:
            clear_chunk_cache(st.session_state.get("train_year_active", train_year))
        for key in ["train_schedule", "train_year_active", "train_fit_done",
                    "train_result"]:
            st.session_state.pop(key, None)
        st.rerun()

    # ── Start / Resume retrain ────────────────────────────────────────────────
    if train_btn:
        # If starting fresh or switching year, rebuild schedule
        if ("train_schedule" not in st.session_state
                or st.session_state.get("train_year_active") != train_year):
            st.session_state["train_schedule"]    = season_chunk_schedule(train_year)
            st.session_state["train_year_active"] = train_year
            st.session_state.pop("train_fit_done", None)
            st.session_state.pop("train_result",   None)
        st.session_state["train_running"] = True
        st.rerun()

    # ── Streaming download + train loop ──────────────────────────────────────
    if st.session_state.get("train_running") and \
            not st.session_state.get("train_fit_done"):

        sched = st.session_state.get("train_schedule", [])
        if not sched:
            st.error("No schedule — click Start first.")
            st.session_state.pop("train_running", None)
        else:
            # Find the next chunk that hasn't been downloaded yet
            pending = [c for c in sched if not c["done"]]

            if pending:
                item   = pending[0]
                pbar   = st.progress(0.0)
                status = st.empty()
                n_done = sum(1 for c in sched if c["done"])
                n_tot  = len(sched)

                status.text(
                    f"Downloading {item['start'].strftime('%b %d')} – "
                    f"{item['end'].strftime('%b %d, %Y')} "
                    f"({n_done+1}/{n_tot})…")
                pbar.progress((n_done) / n_tot)

                try:
                    import time as _time
                    chunk_df = fetch_chunk(
                        st.session_state["train_year_active"],
                        item["start"], item["end"], item["cache"])
                    item["done"] = True
                    _time.sleep(1.2)  # be respectful to Savant
                except Exception as e:
                    st.error(f"Chunk download failed: {e}")
                    st.session_state.pop("train_running", None)
                    st.stop()

                pbar.progress((n_done + 1) / n_tot)
                status.text(f"Chunk {n_done+1}/{n_tot} done. Refreshing…")
                st.rerun()   # come back for the next chunk

            else:
                # All chunks downloaded — now fit the model
                status = st.empty()
                pbar   = st.progress(0.0)
                status.text("All data downloaded. Fitting model (this takes ~2 min)…")

                def _fit_progress(msg, pct):
                    status.text(msg)
                    pbar.progress(min(float(pct), 1.0))

                try:
                    loaded_chunks = []
                    for item in sched:
                        if item["cache"].exists():
                            try:
                                loaded_chunks.append(pd.read_parquet(item["cache"]))
                            except Exception:
                                pass

                    if not loaded_chunks:
                        st.error("No cached chunks found — try resetting and restarting.")
                        st.session_state.pop("train_running", None)
                        st.stop()

                    result = train_model_streaming(loaded_chunks, status_fn=_fit_progress)
                    st.session_state["train_fit_done"] = True
                    st.session_state["train_result"]   = result
                    st.session_state.pop("train_running", None)
                    pbar.progress(1.0)
                    st.rerun()

                except Exception as e:
                    st.error(f"Model fitting failed: {e}")
                    import traceback; st.code(traceback.format_exc())
                    st.session_state.pop("train_running", None)

    # ── Show result after successful train ────────────────────────────────────
    if st.session_state.get("train_fit_done") and \
            "train_result" in st.session_state:
        r = st.session_state["train_result"]
        st.success(
            f"✅ Model trained on {r['n_pitches']:,} pitches! "
            f"P50={r['score_p50']} · P95={r['score_p95']} · P99={r['score_p99']}")

    # Pitcher lookup
    if load_btn and pitcher_input and selected_pitcher_id:
        if not model_is_trained():
            st.error("No model loaded. Place pickle files in data/model/ or retrain.")
        else:
            pid = selected_pitcher_id
            with st.spinner(f"Fetching {lookup_year} data for {pitcher_input} (ID: {pid})…"):
                if days_back:
                    pdf = fetch_pitcher_live(pid, days_back=days_back)
                else:
                    pdf = fetch_pitcher_season(pid, lookup_year)

            if len(pdf) == 0:
                st.error("No data returned. Season may not have started yet.")
            else:
                st.success(f"{len(pdf):,} pitches loaded")
                try:
                    summary, label = process_and_score(pdf, pid)
                    render_card(summary, label)
                except Exception as e:
                    st.error(f"Processing error: {e}")
                    import traceback; st.code(traceback.format_exc())

    elif load_btn and not pitcher_input:
        st.warning("Please search for and select a pitcher first.")

    elif not load_btn and not st.session_state.get("train_running") \
            and not st.session_state.get("train_fit_done"):
        render_landing()

elif data_mode == "Upload CSV":
    if uploaded_file is not None:
        if not model_is_trained():
            st.error("No model loaded. Place pickle files in data/model/ or retrain.")
        else:
            raw = pd.read_csv(uploaded_file, low_memory=False)
            st.success(f"{len(raw):,} rows loaded")
            try:
                scored = score_from_csv(raw)
                if 'pitcher' in scored.columns and 'player_name' in scored.columns:
                    pitchers = (scored.groupby(['pitcher', 'player_name'])
                                .size().reset_index(name='n')
                                .sort_values('n', ascending=False))
                    opts = {f"{r['player_name']} ({r['n']}p)": r['pitcher']
                            for _, r in pitchers.head(50).iterrows()}
                    sel = st.selectbox("Select Pitcher", list(opts.keys()))
                    pid = opts[sel]
                else:
                    pid = None
                summary = build_pitcher_summary(scored, pid)
                render_card(summary, f"Physics {model_trainer.PHYSICS_VERSION}")
            except Exception as e:
                st.error(f"Processing error: {e}")
                import traceback; st.code(traceback.format_exc())
    else:
        render_landing()
