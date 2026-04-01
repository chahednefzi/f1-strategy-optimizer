import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="F1 Strategy Optimizer",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
    font-family: 'Inter', sans-serif;
    font-size: 7.2rem;
    font-weight: 600;
    color: #E10600;
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
    letter-spacing: -0.02em;
    line-height: 1.1;
    }   

    .subtitle-bar {
        background-color: #D3D3D3 ;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        margin-bottom: 1.8rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }

    .subtitle-bar span {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 500;
        color: #444;
        letter-spacing: 0.01em;
    }

    .subtitle-bar .separator {
        color: #aaa;
        margin: 0 0.6rem;
    }

    .era-badge-old {
        background-color: #1E41FF;
        color: white;
        padding: 0.15rem 0.55rem;
        border-radius: 0.25rem;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    .era-badge-new {
        background-color: #E10600;
        color: white;
        padding: 0.15rem 0.55rem;
        border-radius: 0.25rem;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
    }

    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .stMetric label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        color: #555;
    }

    .stMetric [data-testid="metric-container"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    .stSidebar [data-testid="stSidebarContent"] {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODEL V2
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model     = joblib.load('models/xgboost_v2_with2026.pkl')
    le_driver = joblib.load('models/le_driver_v2.pkl')
    le_race   = joblib.load('models/le_race_v2.pkl')
    with open('models/model_metadata_v2.json') as f:
        metadata = json.load(f)
    return model, le_driver, le_race, metadata

model, le_driver, le_race, metadata = load_model()
features     = metadata['features']
compound_map = metadata['compound_map']

# ─────────────────────────────────────────
# CIRCUIT BASELINES (depuis CSV)
# ─────────────────────────────────────────
@st.cache_data
def load_baselines():
    df = pd.read_csv('data/all_laps_2024_2025_2026.csv')
    return (
        df.groupby(['Race', 'Year'])['LapTime']
        .median()
        .reset_index()
        .rename(columns={'LapTime': 'CircuitBaseline'})
    )

circuit_baselines = load_baselines()

def get_baseline(race_name, year):
    match = circuit_baselines[
        (circuit_baselines['Race'] == race_name) &
        (circuit_baselines['Year'] == year)
    ]
    if len(match) == 0:
        match = circuit_baselines[
            circuit_baselines['Race'] == race_name
        ].sort_values('Year', ascending=False)
    return float(match.iloc[0]['CircuitBaseline'])

# ─────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────
def predict_lap_time(driver, compound, tire_age, lap_number,
                     race_name, year, position=5, total_laps=55):
    circuit_baseline = get_baseline(race_name, year)
    regulation_era   = 0 if year <= 2025 else 1
    compound_encoded = compound_map[compound]

    if driver in metadata['driver_classes']:
        driver_encoded = metadata['driver_classes'].index(driver)
    else:
        driver_encoded = 0

    if race_name in metadata['race_classes']:
        race_encoded = metadata['race_classes'].index(race_name)
    else:
        race_encoded = 0

    input_data = pd.DataFrame([{
        'CircuitBaseline'        : circuit_baseline,
        'TyreLife'               : tire_age,
        'TyreLifeSquared'        : tire_age ** 2,
        'CompoundEncoded'        : compound_encoded,
        'CompoundTireInteraction': compound_encoded * tire_age,
        'IsNewTire'              : 1 if tire_age <= 5 else 0,
        'StintProgress'          : min(tire_age / 30, 1.0),
        'DriverEncoded'          : driver_encoded,
        'FuelEffect'             : (total_laps - lap_number) * 0.035,
        'Position'               : position,
        'LapNumber'              : lap_number,
        'RaceEncoded'            : race_encoded,
        'RegulationEra'          : regulation_era
    }])[features]

    delta    = model.predict(input_data)[0]
    lap_time = circuit_baseline + delta
    return round(float(lap_time), 3)

# ─────────────────────────────────────────
# SIMULATOR
# ─────────────────────────────────────────
class RaceStrategySimulator:
    def __init__(self, race_name, year, total_laps=55, pit_stop_time=22.0):
        self.race_name     = race_name
        self.year          = year
        self.total_laps    = total_laps
        self.pit_stop_time = pit_stop_time
        self.baseline      = get_baseline(race_name, year)

    def simulate_strategy(self, driver, strategy, position=5):
        total_time  = 0
        current_lap = 1
        all_lap_times = []
        stint_details = []

        for i, stint in enumerate(strategy):
            stint_laps = []
            for tire_age in range(1, stint['laps'] + 1):
                lt = predict_lap_time(
                    driver, stint['compound'], tire_age,
                    current_lap, self.race_name, self.year,
                    position, self.total_laps
                )
                stint_laps.append(lt)
                all_lap_times.append({
                    'lap'     : current_lap,
                    'time'    : lt,
                    'compound': stint['compound'],
                    'tire_age': tire_age,
                    'stint'   : i + 1
                })
                current_lap += 1

            stint_time = sum(stint_laps)
            total_time += stint_time
            stint_details.append({
                'stint'   : i + 1,
                'compound': stint['compound'],
                'laps'    : stint['laps'],
                'time'    : stint_time,
                'avg'     : stint_time / stint['laps']
            })

            if i < len(strategy) - 1:
                total_time += self.pit_stop_time

        return {
            'total_time'    : total_time,
            'total_minutes' : total_time / 60,
            'lap_times'     : all_lap_times,
            'stints'        : stint_details,
            'num_pit_stops' : len(strategy) - 1
        }

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown(
    '<p class="main-header">🏎️ F1 Race Strategy Optimizer</p>',
    unsafe_allow_html=True
)

st.markdown(f"""
<div class="subtitle-bar">
    <span>
        XGBoost ML Model
        <span class="separator">|</span>
        R² = {metadata['performance']['R2']}
        <span class="separator">|</span>
        MAE = {metadata['performance']['MAE']}s
        <span class="separator">|</span>
        <span class="era-badge-old">2024–2025</span>
        &nbsp;+&nbsp;
        <span class="era-badge-new">2026 NEW REGS</span>
    </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.header(" Configuration")

# Année / Ère
year = st.sidebar.radio(
    "📅 Season",
    [2024, 2025, 2026],
    index=2,
    horizontal=True
)

era_label = "🔴 2026 NEW REGS" if year == 2026 else "🔵 V6 Hybrid Era"
st.sidebar.markdown(f"**Era:** {era_label}")

# Circuits disponibles pour cette année
available_circuits = sorted(
    circuit_baselines[
        circuit_baselines['Year'] == year
    ]['Race'].unique()
)

if len(available_circuits) == 0:
    available_circuits = sorted(
        circuit_baselines['Race'].unique()
    )

circuit = st.sidebar.selectbox("🏁 Circuit", available_circuits)
baseline_val = get_baseline(circuit, year)
st.sidebar.metric("Circuit Baseline", f"{baseline_val:.2f}s")

# Pilote
drivers = sorted(metadata['driver_classes'])
driver  = st.sidebar.selectbox(
    "👤 Driver",
    drivers,
    index=drivers.index('NOR') if 'NOR' in drivers else 0
)

total_laps = st.sidebar.number_input(
    " Total Laps", min_value=40, max_value=78, value=57
)
position = st.sidebar.slider(
    "📍 Average Position", min_value=1, max_value=20, value=3
)
pit_stop_time = st.sidebar.slider(
    " Pit Stop Loss (s)", min_value=18.0, max_value=30.0,
    value=22.0, step=0.5
)

st.sidebar.markdown("---")
st.sidebar.markdown("###  Model Info")
st.sidebar.metric("R²",    metadata['performance']['R2'])
st.sidebar.metric("MAE",   f"{metadata['performance']['MAE']}s")
st.sidebar.metric("Laps 2024-25",
                  f"{metadata['dataset']['laps_2024_25']:,}")
st.sidebar.metric("Laps 2026",
                  f"{metadata['dataset']['laps_2026']:,}")

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Quick Prediction",
    "🏁 Strategy Comparison",
    "🔍 Pit Window Optimizer",
    "📈 Model Performance"
])

# ══════════════════════════════════════════
# TAB 1 — QUICK PREDICTION
# ══════════════════════════════════════════
with tab1:
    st.header(" Single Lap Time Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        compound_q  = st.selectbox("Tire Compound",
                                   ['SOFT', 'MEDIUM', 'HARD'])
        tire_age_q  = st.slider("Tire Age (laps)", 1, 45, 10)
    with col2:
        lap_num_q   = st.slider("Current Lap", 1, total_laps, 25)
        position_q  = st.slider("Position", 1, 20, position,
                                 key='pos_q')
    with col3:
        st.markdown(f"""
        <div style="font-family:'Inter',sans-serif;">
        <p style="font-size:0.75rem; color:#FFFFFF; margin:0 0 0.1rem 0; font-weight:500;">Circuit</p>
        <p style="font-size:0.9rem; color:#FFFFFF; margin:0 0 1rem 0; font-weight:600;">{circuit}</p>

        <p style="font-size:0.75rem; color:#FFFFFF; margin:0 0 0.1rem 0; font-weight:500;">Baseline</p>
        <p style="font-size:0.9rem; color:#FFFFFF; margin:0 0 1rem 0; font-weight:600;">{baseline_val:.2f}s</p>

        <p style="font-size:0.75rem; color:#FFFFFF; margin:0 0 0.1rem 0; font-weight:500;">Season</p>
        <p style="font-size:0.9rem; color:#FFFFFF; margin:0 0 0 0; font-weight:600;">
                {year} — {'NEW REGS' if year==2026 else 'V6 Era'}
        </p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔮 Predict Lap Time", type="primary"):
        pred = predict_lap_time(
            driver, compound_q, tire_age_q, lap_num_q,
            circuit, year, position_q, total_laps
        )
        delta = pred - baseline_val

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Lap Time", f"{pred:.3f}s")
        c2.metric("Delta from Baseline",
                  f"{delta:+.3f}s",
                  delta_color="inverse")
        c3.metric("Compound", compound_q)
        c4.metric("Tire Age", f"{tire_age_q} laps")

        # Mini chart — dégradation sur 30 tours
        st.markdown("#### 📉 Degradation Preview")
        ages  = list(range(1, 31))
        times = [
            predict_lap_time(driver, compound_q, a, lap_num_q,
                             circuit, year, position_q, total_laps)
            for a in ages
        ]

        compound_colors = {
            'SOFT': '#E10600', 'MEDIUM': '#FFD700', 'HARD': '#CCCCCC'
        }
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ages, y=times,
            mode='lines+markers',
            line=dict(color=compound_colors[compound_q], width=3),
            marker=dict(size=6),
            name=f'{compound_q} degradation'
        ))
        fig.add_hline(
            y=baseline_val, line_dash="dash",
            line_color="gray", annotation_text="Circuit Baseline"
        )
        fig.update_layout(
            title=f"Tire Degradation — {compound_q} — {circuit} {year}",
            xaxis_title="Tire Age (laps)",
            yaxis_title="Lap Time (s)",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════
# TAB 2 — STRATEGY COMPARISON
# ══════════════════════════════════════════
with tab2:
    st.header("🏁 Race Strategy Comparison")

    num_strategies = st.number_input(
        "Number of strategies", 2, 5, 3
    )

    strategies = {}
    cols = st.columns(num_strategies)

    for i in range(num_strategies):
        with cols[i]:
            st.markdown(f"### Strategy {i+1}")
            name       = st.text_input("Name", f"Strategy {i+1}",
                                        key=f"name_{i}")
            num_stints = st.number_input("Pit Stops", 1, 3, 1,
                                          key=f"pits_{i}")
            stints     = []
            remaining  = total_laps

            for j in range(num_stints + 1):
                st.markdown(f"**Stint {j+1}**")
                cpd = st.selectbox(
                    "Compound", ['SOFT', 'MEDIUM', 'HARD'],
                    key=f"cpd_{i}_{j}"
                )
                if j < num_stints:
                    laps_s = st.number_input(
                        "Laps", 1,
                        remaining - (num_stints - j),
                        min(25, remaining - (num_stints - j)),
                        key=f"laps_{i}_{j}"
                    )
                else:
                    laps_s = remaining
                stints.append({'compound': cpd, 'laps': laps_s})
                remaining -= laps_s

            strategies[name] = stints

    if st.button(" Run Simulation", type="primary"):
        sim = RaceStrategySimulator(
            circuit, year, total_laps, pit_stop_time
        )
        rows = []
        lap_time_traces = {}

        for name, strat in strategies.items():
            res = sim.simulate_strategy(driver, strat, position)
            rows.append({
                'Strategy'        : name,
                'Compounds'       : ' → '.join(
                    s['compound'] for s in strat),
                'Pit Stops'       : res['num_pit_stops'],
                'Total Time (min)': round(res['total_minutes'], 2),
                'Total Time (s)'  : round(res['total_time'],    1)
            })
            lap_time_traces[name] = res['lap_times']

        df_res = pd.DataFrame(rows).sort_values('Total Time (s)')
        df_res['Delta (s)'] = (
            df_res['Total Time (s)'] -
            df_res['Total Time (s)'].iloc[0]
        ).round(1)

        best = df_res.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric(" Best Strategy", best['Strategy'])
        c2.metric("⏱ Time", f"{best['Total Time (min)']:.2f} min")
        c3.metric(" Pit Stops", int(best['Pit Stops']))

        st.dataframe(df_res.drop(columns=['Total Time (s)']),
                     use_container_width=True, hide_index=True)

        # Chart 1: Bar comparison
        palette = ['#E10600','#15151E','#1E41FF','#FF8700','#00A550']
        fig1 = go.Figure()
        for idx, (_, row) in enumerate(df_res.iterrows()):
            fig1.add_trace(go.Bar(
                x=[row['Strategy']],
                y=[row['Total Time (min)']],
                name=row['Strategy'],
                marker_color=palette[idx % len(palette)],
                text=f"{row['Total Time (min)']:.2f} min",
                textposition='outside'
            ))
        fig1.update_layout(
            title="Strategy Comparison — Total Race Time",
            yaxis_title="Total Time (minutes)",
            height=400, showlegend=False
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Lap time evolution
        fig2 = go.Figure()
        for idx, (name, laps) in enumerate(lap_time_traces.items()):
            df_laps = pd.DataFrame(laps)
            fig2.add_trace(go.Scatter(
                x=df_laps['lap'],
                y=df_laps['time'],
                mode='lines',
                name=name,
                line=dict(color=palette[idx % len(palette)],
                          width=2.5)
            ))
        fig2.update_layout(
            title="Lap Time Evolution by Strategy",
            xaxis_title="Lap",
            yaxis_title="Lap Time (s)",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════
# TAB 3 — PIT WINDOW OPTIMIZER
# ══════════════════════════════════════════
with tab3:
    st.header(" Pit Window Optimization")

    col1, col2 = st.columns(2)
    with col1:
        c1_opt = st.selectbox("First Compound",
                               ['SOFT','MEDIUM','HARD'], index=1,
                               key='c1_opt')
    with col2:
        c2_opt = st.selectbox("Second Compound",
                               ['SOFT','MEDIUM','HARD'], index=2,
                               key='c2_opt')

    col3, col4 = st.columns(2)
    with col3:
        s_min = st.slider("Search Min Lap", 10, 35, 18)
    with col4:
        s_max = st.slider("Search Max Lap", s_min+5, 45, 38)

    st.info(f"Testing {s_max - s_min + 1} pit stop windows")

    if st.button(" Find Optimal Pit Window", type="primary"):
        with st.spinner("Optimizing..."):
            sim  = RaceStrategySimulator(
                circuit, year, total_laps, pit_stop_time
            )
            rows = []
            prog = st.progress(0)
            total_iter = s_max - s_min + 1

            for idx, pit_lap in enumerate(range(s_min, s_max + 1)):
                strat = [
                    {'compound': c1_opt, 'laps': pit_lap},
                    {'compound': c2_opt,
                     'laps': total_laps - pit_lap}
                ]
                res = sim.simulate_strategy(driver, strat, position)
                rows.append({
                    'pit_lap'      : pit_lap,
                    'total_time'   : res['total_time'],
                    'total_minutes': res['total_minutes']
                })
                prog.progress((idx + 1) / total_iter)

            prog.empty()
            df_opt  = pd.DataFrame(rows)
            optimal = df_opt.loc[df_opt['total_time'].idxmin()]
            worst   = df_opt.loc[df_opt['total_time'].idxmax()]

            threshold = optimal['total_time'] + 1.0
            opt_range = df_opt[df_opt['total_time'] <= threshold]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🏆 Optimal Pit Lap",
                      f"Lap {int(optimal['pit_lap'])}")
            c2.metric("⏱ Total Time",
                      f"{optimal['total_minutes']:.2f} min")
            c3.metric(" Time Saved",
                      f"{worst['total_time']-optimal['total_time']:.1f}s")
            c4.metric(" Flexible Window",
                      f"Laps {int(opt_range['pit_lap'].min())}"
                      f"–{int(opt_range['pit_lap'].max())}")

            # Chart
            colors_pit = []
            min_t = df_opt['total_time'].min()
            for t in df_opt['total_time']:
                if t == min_t:          colors_pit.append('green')
                elif t <= min_t + 1:   colors_pit.append('gold')
                elif t <= min_t + 3:   colors_pit.append('orange')
                else:                  colors_pit.append('red')

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_opt['pit_lap'], y=df_opt['total_minutes'],
                mode='lines+markers',
                line=dict(color='steelblue', width=3),
                marker=dict(size=10, color=colors_pit,
                            line=dict(width=2, color='white')),
                hovertemplate='Lap %{x}<br>%{y:.2f} min<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[optimal['pit_lap']],
                y=[optimal['total_minutes']],
                mode='markers',
                marker=dict(size=22, color='gold',
                            symbol='star',
                            line=dict(width=2, color='black')),
                name='Optimal'
            ))
            fig.update_layout(
                title=f"Pit Window: {c1_opt} → {c2_opt} "
                      f"— {circuit} {year}",
                xaxis_title="Pit Stop Lap",
                yaxis_title="Total Race Time (min)",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════
with tab4:
    st.header(" Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score",  metadata['performance']['R2'])
    c2.metric("MAE",       f"{metadata['performance']['MAE']}s")
    c3.metric("RMSE",      f"{metadata['performance']['RMSE']}s")
    c4.metric("Model",     "XGBoost v2")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("###  Dataset")
        st.metric("Total laps",    f"{metadata['dataset']['total_laps']:,}")
        st.metric("2024-2025 laps",f"{metadata['dataset']['laps_2024_25']:,}")
        st.metric("2026 laps",     f"{metadata['dataset']['laps_2026']:,}")

        st.markdown("### 🔧 Regulation Eras")
        st.markdown("""
        | Era | Years | Flag |
        |-----|-------|------|
        | V6 Hybrid | 2024–2025 | `0` |
        | New Regs  | 2026      | `1` |
        """)

    with col2:
        st.markdown("### + Strengths")
        st.markdown("""
        - R² = 0.9955 across 3 seasons
        - Handles regulation changes via `RegulationEra`
        - Circuit normalization removes track bias
        - 47,000+ laps training data
        - XGBoost — industry standard
        """)
        st.markdown("### - Limitations")
        st.markdown("""
        - 2026 only 3 races (limited data)
        - No real-time weather
        - No safety car prediction
        - Fuel load is approximated
        """)

    st.markdown("---")
    st.markdown("###  Features")
    feat_df = pd.DataFrame({
        'Feature'    : features,
        'Description': [
            'Circuit median lap time (normalization)',
            'Tire age in laps',
            'Tire age² (non-linear degradation)',
            'Compound: SOFT=0, MEDIUM=1, HARD=2',
            'Compound × tire age interaction',
            'New tire flag (age ≤ 5)',
            'Stint progress 0→1',
            'Driver encoding',
            'Fuel load effect',
            'Track position',
            'Lap number',
            'Circuit encoding',
            '🆕 0=2024-25, 1=2026 new regs'
        ]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; padding:1rem;">
    <p>F1 Race Strategy Optimizer v2 &nbsp;|&nbsp;
       XGBoost ML &nbsp;|&nbsp;
       2024 + 2025 + 2026 Data</p>
    <p>FastF1 API &nbsp;|&nbsp; R²=0.9955 &nbsp;|&nbsp;
       MAE=0.422s</p>
</div>
""", unsafe_allow_html=True)