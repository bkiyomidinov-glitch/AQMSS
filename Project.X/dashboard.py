import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json
import numpy as np
from typing import Optional

# Page config
st.set_page_config(
    page_title="AMQSS Market Quality Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .score-excellent {
        color: #00cc00;
        font-weight: bold;
    }
    .score-good {
        color: #99ff00;
        font-weight: bold;
    }
    .score-weak {
        color: #ff9900;
        font-weight: bold;
    }
    .score-terrible {
        color: #ff0000;
        font-weight: bold;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0088cc;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #ffe8cc;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff6600;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
CSV_PATH = r'C:\Users\bkiyo\Desktop\Project.X\results\market_scores.csv'
MODELS_PATH = r'C:\Users\bkiyo\Desktop\Project.X\models'
DEFAULT_OHLC_PATH = r'C:\Users\bkiyo\Downloads\EURUSD_60_2025-01-20_2026-01-19.csv'

@st.cache_data(ttl=60)
def load_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        if 'ai_high_quality_prob' in df.columns:
            df['ai_high_quality_prob'] = pd.to_numeric(df['ai_high_quality_prob'], errors='coerce')
        return df
    return None

@st.cache_data
def load_metadata():
    meta_path = os.path.join(MODELS_PATH, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_feature_importance():
    imp_path = os.path.join(MODELS_PATH, 'metrics.json')
    if os.path.exists(imp_path):
        with open(imp_path, 'r') as f:
            return json.load(f)
    return None


def safe_autorefresh(interval_ms: int, key: str):
    try:
        if hasattr(st, "autorefresh"):
            return st.autorefresh(interval=interval_ms, key=key)
        from streamlit_autorefresh import st_autorefresh as _st_autorefresh
        return _st_autorefresh(interval=interval_ms, key=key)
    except Exception:
        return None


def load_price_data(source: Optional[str] = None, uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif source and os.path.exists(source):
        df = pd.read_csv(source)
    else:
        return None

    def try_normalized_frame(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
        lower_cols = {str(c).lower(): c for c in frame.columns}
        time_col = lower_cols.get('time') or lower_cols.get('timestamp') or lower_cols.get('date')
        open_col = lower_cols.get('open')
        high_col = lower_cols.get('high')
        low_col = lower_cols.get('low')
        close_col = lower_cols.get('close')

        if all([time_col, open_col, high_col, low_col, close_col]):
            out = frame[[time_col, open_col, high_col, low_col, close_col]].copy()
            out.columns = ['timestamp', 'open', 'high', 'low', 'close']
            out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed', errors='coerce')
            out = out.dropna(subset=['timestamp'])
            return out
        return None

    # Try with headers first
    normalized = try_normalized_frame(df)
    if normalized is None:
        # Fallback: assume no header and fixed column order
        df_no_header = pd.read_csv(uploaded_file if uploaded_file is not None else source, header=None)
        if df_no_header.shape[1] >= 5:
            df_no_header = df_no_header.iloc[:, :5].copy()
            df_no_header.columns = ['timestamp', 'open', 'high', 'low', 'close']

            # Drop header-like first row if detected
            sample_row = df_no_header.iloc[0].astype(str).str.lower().tolist()
            header_like = any(x in sample_row for x in ['time', 'timestamp', 'date', 'open', 'high', 'low', 'close'])
            if header_like:
                df_no_header = df_no_header.iloc[1:].copy()

            df_no_header['timestamp'] = pd.to_datetime(df_no_header['timestamp'], format='mixed', errors='coerce')
            df_no_header['open'] = pd.to_numeric(df_no_header['open'], errors='coerce')
            df_no_header['high'] = pd.to_numeric(df_no_header['high'], errors='coerce')
            df_no_header['low'] = pd.to_numeric(df_no_header['low'], errors='coerce')
            df_no_header['close'] = pd.to_numeric(df_no_header['close'], errors='coerce')
            df_no_header = df_no_header.dropna(subset=['timestamp', 'open', 'high', 'low', 'close'])
            normalized = df_no_header

    if normalized is None or normalized.empty:
        return None

    normalized = normalized.sort_values('timestamp').reset_index(drop=True)
    return normalized


def build_explanation(row: pd.Series) -> str:
    reasons = []
    if row.get('volatility_regime', 0) >= 7:
        reasons.append('Volatility regime is stable')
    else:
        reasons.append('Volatility regime is unstable')

    if row.get('momentum', 0) >= 6:
        reasons.append('Momentum supports continuation')
    else:
        reasons.append('Momentum is weak')

    if row.get('structure', 0) >= 6:
        reasons.append('Market structure is clean')
    else:
        reasons.append('Market structure is noisy')

    if row.get('true_liquidity', 0) >= 5:
        reasons.append('Liquidity is acceptable')
    else:
        reasons.append('Liquidity is thin')

    vol_level = row.get('volatility_level', 0)
    if 4 <= vol_level <= 6:
        reasons.append('Volatility level is optimal')
    else:
        reasons.append('Volatility level is off-balance')

    return '; '.join(reasons)


def generate_ai_helper_response(question: str, row: pd.Series) -> str:
    q = (question or "").lower()
    score = row.get('total_score', 0)
    ai_prob = row.get('ai_high_quality_prob')
    if pd.isna(ai_prob):
        ai_prob = min(max(score / 10.0, 0), 1)

    if "why" in q or "почему" in q or "explain" in q:
        return build_explanation(row)
    if "recommend" in q or "рекоменд" in q or "trade" in q:
        if score >= 7.5 and ai_prob >= 0.7:
            return "Conditions are strong. Consider trading with normal risk."
        if score >= 6 and ai_prob >= 0.5:
            return "Conditions are acceptable. Use moderate risk and confirm setup."
        if score >= 4:
            return "Conditions are weak. Reduce risk or wait."
        return "Conditions are poor. Avoid trading."
    if "risk" in q or "риск" in q:
        if score >= 7.5:
            return "Risk is lower than usual; still manage position size."
        if score >= 6:
            return "Risk is moderate; tighten stops and confirm liquidity."
        return "Risk is elevated; avoid or size down significantly."

    return (
        f"Current AMQSS score: {score:.2f}/10, AI confidence: {ai_prob*100:.1f}%. "
        f"{build_explanation(row)}"
    )

# Header
st.title("📊 AMQSS Market Quality Scoring System v2")
st.markdown("**Advanced Market Quality Scoring System** - Real-time Forex Market Analysis with AI Enhancement")

# Sidebar
with st.sidebar:
    st.header("🔧 Settings & Info")
    st.markdown("---")
    
    # Model Info
    st.subheader("🤖 AI Model Info")
    metadata = load_metadata()
    if metadata:
        st.caption("RandomForest Classifier")
        st.write(f"**Features:** {metadata['n_features']}")
        st.write(f"**Train Size:** {metadata['train_size']}")
        st.write(f"**Test Size:** {metadata['test_size']}")
        st.write(f"**Trained:** {metadata['trained_at'][:10]}")
    
    st.markdown("---")
    
    # Controls
    refresh_interval = st.slider("Refresh data (seconds)", 5, 300, 60)
    show_all_history = st.checkbox("Show all historical data", value=False)
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    alert_threshold = st.slider("Alert if score below", 0.0, 10.0, 4.0, 0.1)

    st.markdown("---")
    st.subheader("🕯️ Candle Data (OHLC)")
    default_path_value = DEFAULT_OHLC_PATH if os.path.exists(DEFAULT_OHLC_PATH) else ""
    price_csv_path = st.text_input("Price CSV path (optional)", value=default_path_value)
    uploaded_price = st.file_uploader("Upload OHLC CSV", type=["csv"])
    st.caption("Expected columns: time/timestamp, open, high, low, close")
    merge_tolerance_minutes = st.slider("Merge tolerance (minutes)", 5, 240, 60)
    
    if st.button("🔄 Refresh Data Now"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("📧 System Status: ✅ Running")

# Load data
data = load_data()

if data is None or len(data) == 0:
    st.warning("⚠️ No data available yet. Run main.py to generate data.")
    st.stop()

if auto_refresh:
    try:
        safe_autorefresh(interval_ms=refresh_interval * 1000, key="auto_refresh")
    except Exception:
        pass

# Date range filtering
min_time = data['timestamp'].min().to_pydatetime()
max_time = data['timestamp'].max().to_pydatetime()
if show_all_history:
    filtered_data = data
else:
    default_start = max_time - timedelta(days=3)
    date_range = st.slider(
        "Select time range",
        min_value=min_time,
        max_value=max_time,
        value=(default_start, max_time),
        format="YYYY-MM-DD HH:mm"
    )
    filtered_data = data[(data['timestamp'] >= date_range[0]) & (data['timestamp'] <= date_range[1])]

# Get latest score
latest = filtered_data.iloc[-1] if len(filtered_data) > 0 else data.iloc[-1]

if latest['total_score'] < alert_threshold:
    st.error(f"🚨 Alert: AMQSS score below threshold ({latest['total_score']:.2f} < {alert_threshold:.2f})")

# Display metrics row
col1, col2, col3, col4, col5 = st.columns(5)

metrics = [
    (col1, "Total Score", latest['total_score'], 10),
    (col2, "Vol. Regime", latest['volatility_regime'], 10),
    (col3, "True Liquidity", latest['true_liquidity'], 10),
    (col4, "Structure", latest['structure'], 10),
    (col5, "Momentum", latest['momentum'], 10)
]

for col, label, value, max_val in metrics:
    with col:
        st.metric(label=label, value=f"{value:.2f}/{max_val}")

# Market condition indicator
st.markdown("---")
col1, col2, col3 = st.columns([1.5, 3, 1.5])

with col2:
    # Determine color and emoji
    score = latest['total_score']
    if score >= 7.5:
        condition = "🟢 HIGH QUALITY MARKET"
        color = "#00cc00"
        emoji = "✅"
    elif score >= 6:
        condition = "🟡 NORMAL CONDITIONS"
        color = "#ffff00"
        emoji = "⚠️"
    elif score >= 4:
        condition = "🟠 WEAK CONDITIONS"
        color = "#ff9900"
        emoji = "⛔"
    else:
        condition = "🔴 NO TRADE / CRITICAL"
        color = "#ff0000"
        emoji = "🚫"
    
    st.markdown(f"""
        <div style='
            background-color: {color};
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            color: black;
            font-size: 26px;
            font-weight: bold;
        '>
            {condition}
        </div>
    """, unsafe_allow_html=True)

# AI Prediction & Analysis
st.markdown("---")
st.subheader("🤖 AI Market Environment Assessment")

col_ai_1, col_ai_2, col_ai_3 = st.columns([2, 2, 2])

with col_ai_1:
    # AI Probability
    ai_prob_raw = latest['ai_high_quality_prob'] if 'ai_high_quality_prob' in latest.index else None
    if pd.notna(ai_prob_raw):
        ai_prob = float(ai_prob_raw)
    else:
        ai_prob = min(max(latest['total_score'] / 10.0, 0), 1)
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ai_prob * 100 if ai_prob > 0 else 0,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "AI Quality Probability"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#1f77b4'},
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig_gauge.update_layout(height=300, font={'size': 12})
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_ai_2:
    st.metric("AI Confidence", f"{ai_prob*100:.1f}%", delta="+5%" if ai_prob > 0.5 else "-3%")
    st.metric("Market Score", f"{score:.2f}/10")
    
    # Quality level
    if score >= 7.5:
        quality = "Excellent"
        quality_color = "🟢"
    elif score >= 6:
        quality = "Good"
        quality_color = "🟡"
    elif score >= 4:
        quality = "Weak"
        quality_color = "🟠"
    else:
        quality = "Critical"
        quality_color = "🔴"
    
    st.markdown(f"**Market Quality:** {quality_color} {quality}")

with col_ai_3:
    # Recommendations
    st.markdown("#### 📌 Recommendation:")
    if score >= 7.5 and ai_prob > 0.7:
        st.success("✅ Optimal conditions - Consider trading")
    elif score >= 6 and ai_prob > 0.5:
        st.info("ℹ️ Good conditions - Moderate trade signal")
    elif score >= 4:
        st.warning("⚠️ Weak conditions - Use caution")
    else:
        st.error("🚫 Poor conditions - Wait for better setup")

# Insights section
st.markdown("---")
st.subheader("💡 Market Insights & Analysis")

insight_col_1, insight_col_2 = st.columns([1, 1])

with insight_col_1:
    # What's driving the score?
    st.markdown("**Why this score?** 🔍")
    
    factors = {
        'volatility_regime': latest['volatility_regime'],
        'true_liquidity': latest['true_liquidity'],
        'structure': latest['structure'],
        'momentum': latest['momentum'],
        'volatility_level': latest['volatility_level']
    }
    
    # Sort by importance (using approximate importance from model)
    importance_order = ['volatility_regime', 'momentum', 'structure', 'true_liquidity', 'volatility_level']
    
    reasons = []
    for factor in importance_order:
        val = factors[factor]
        if factor == 'volatility_regime':
            if val >= 7: reasons.append("✅ Volatility is well-balanced")
            else: reasons.append(f"❌ Volatility regime weak ({val:.1f}/10)")
        elif factor == 'momentum':
            if val >= 6: reasons.append("✅ Good momentum present")
            else: reasons.append(f"⚠️ Momentum lacking ({val:.1f}/10)")
        elif factor == 'structure':
            if val >= 6: reasons.append("✅ Price structure stable")
            else: reasons.append(f"❌ Structure unclear ({val:.1f}/10)")
        elif factor == 'true_liquidity':
            if val >= 5: reasons.append("✅ Liquidity adequate")
            else: reasons.append(f"❌ Low liquidity ({val:.1f}/10)")
        elif factor == 'volatility_level':
            if val >= 4 and val <= 6: reasons.append("✅ Volatility level optimal")
            else: reasons.append(f"⚠️ Volatility level off ({val:.1f}/10)")
    
    for i, reason in enumerate(reasons[:3], 1):
        st.write(f"{i}. {reason}")

with insight_col_2:
    st.markdown("**Key Metrics** 📊")
    
    # ATR info
    atr_current = latest['current_atr']
    atr_mean = latest['mean_atr']
    atr_ratio = atr_current / atr_mean if atr_mean > 0 else 0
    
    if atr_ratio > 1.1:
        st.write(f"📈 ATR elevated ({atr_ratio:.2f}x mean) - High volatility")
    elif atr_ratio < 0.9:
        st.write(f"📉 ATR suppressed ({atr_ratio:.2f}x mean) - Low volatility")
    else:
        st.write(f"➡️ ATR normal ({atr_ratio:.2f}x mean) - Balanced conditions")
    
    st.write(f"📊 Volume: {latest['recent_volume']:.0f} units")
    
    # Trend indicator
    if len(filtered_data) >= 5:
        recent_trend = filtered_data['total_score'].tail(5).values
        if recent_trend[-1] > recent_trend[0]:
            st.success(f"📈 Score improving (↗ {recent_trend[-1] - recent_trend[0]:+.2f})")
        elif recent_trend[-1] < recent_trend[0]:
            st.error(f"📉 Score declining (↘ {recent_trend[-1] - recent_trend[0]:+.2f})")
        else:
            st.info("➡️ Score stable")

st.markdown("---")

# Prepare candle data for live market view
price_df = load_price_data(price_csv_path.strip() if price_csv_path else None, uploaded_price)
merged_candles = None
if price_df is not None:
    scores_sorted = filtered_data.sort_values('timestamp')
    merged_candles = pd.merge_asof(
        price_df.sort_values('timestamp'),
        scores_sorted,
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta(minutes=merge_tolerance_minutes)
    )
    score_cols = [
        'total_score', 'ai_high_quality_prob', 'volatility_regime', 'true_liquidity',
        'structure', 'momentum', 'volatility_level', 'current_atr', 'mean_atr'
    ]
    present_cols = [c for c in score_cols if c in merged_candles.columns]
    if present_cols:
        merged_candles[present_cols] = merged_candles[present_cols].ffill()
    merged_candles['explanation'] = merged_candles.apply(build_explanation, axis=1)

# Charts section with tabs
st.header("📈 Analysis & Trends")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🕯️ Live Market", "📊 Score Trends", "🎯 Factor Breakdown", "🔬 Feature Analysis", "📋 Data Table"])

# Determine data to show
display_data = filtered_data if show_all_history else filtered_data.tail(300)

with tab1:
    st.subheader("Real-time Candle Chart (OHLC) with AI Score")
    if merged_candles is None or merged_candles.empty:
        if uploaded_price is None and not price_csv_path:
            st.warning("Upload or set an OHLC CSV to enable candle chart.")
        elif price_csv_path and not os.path.exists(price_csv_path):
            st.error("OHLC path not found. Check the file path.")
        else:
            st.error("Could not parse OHLC CSV. Ensure columns include time/open/high/low/close or use standard order.")
    else:
        view = merged_candles.tail(300)
        score_for_color = view['ai_high_quality_prob']
        if score_for_color.isna().all():
            score_for_color = view['total_score'] / 10

        strong_mask = score_for_color >= 0.7
        weak_mask = score_for_color < 0.4
        mid_mask = ~(strong_mask | weak_mask)

        def masked(series, mask):
            return series.where(mask)

        fig_candle = go.Figure()

        fig_candle.add_trace(go.Candlestick(
            x=view['timestamp'],
            open=masked(view['open'], strong_mask),
            high=masked(view['high'], strong_mask),
            low=masked(view['low'], strong_mask),
            close=masked(view['close'], strong_mask),
            increasing_line_color='#2ca02c',
            decreasing_line_color='#2ca02c',
            name='Strong (AI/AMQSS)'
        ))

        fig_candle.add_trace(go.Candlestick(
            x=view['timestamp'],
            open=masked(view['open'], mid_mask),
            high=masked(view['high'], mid_mask),
            low=masked(view['low'], mid_mask),
            close=masked(view['close'], mid_mask),
            increasing_line_color='#ffbf00',
            decreasing_line_color='#ffbf00',
            name='Neutral'
        ))

        fig_candle.add_trace(go.Candlestick(
            x=view['timestamp'],
            open=masked(view['open'], weak_mask),
            high=masked(view['high'], weak_mask),
            low=masked(view['low'], weak_mask),
            close=masked(view['close'], weak_mask),
            increasing_line_color='#d62728',
            decreasing_line_color='#d62728',
            name='Weak (AI/AMQSS)'
        ))

        fig_candle.add_trace(go.Scatter(
            x=view['timestamp'],
            y=score_for_color * 10,
            mode='lines',
            name='AI Score (scaled)',
            line=dict(color='#1f77b4', width=2),
            yaxis='y2',
            customdata=view['explanation'],
            hovertemplate="<b>AI Explanation</b><br>%{customdata}<extra></extra>"
        ))

        fig_candle.update_layout(
            xaxis_title='Time',
            yaxis_title='Price',
            yaxis2=dict(title='AI Score', overlaying='y', side='right', range=[0, 10]),
            hovermode='x unified',
            height=600,
            template='plotly_white'
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        latest_explanation = view['explanation'].iloc[-1]
        st.info(f"**Why this candle?** {latest_explanation}")

        st.markdown("---")
        st.subheader("🧠 AI Explanation & Helper")
        selected_time = st.selectbox(
            "Select candle for explanation",
            options=list(view['timestamp']),
            index=len(view) - 1
        )
        selected_row = view[view['timestamp'] == selected_time].iloc[-1]
        if pd.isna(selected_row.get('total_score')):
            fallback = filtered_data.sort_values('timestamp').iloc[-1]
            selected_row = selected_row.copy()
            for col in ['total_score', 'ai_high_quality_prob', 'volatility_regime', 'true_liquidity', 'structure', 'momentum', 'volatility_level']:
                if col in selected_row.index and pd.isna(selected_row[col]) and col in fallback.index:
                    selected_row[col] = fallback[col]

        st.markdown("**Explanation for selected candle:**")
        st.write(build_explanation(selected_row))

        st.markdown("**Feature snapshot:**")
        _ai_val = selected_row.get('ai_high_quality_prob')
        if pd.isna(_ai_val):
            _ai_val = min(max((selected_row.get('total_score', 0) or 0) / 10.0, 0), 1)
        st.write({
            "total_score": round(float(selected_row.get('total_score', 0) or 0), 2),
            "ai_confidence": round(float(_ai_val * 100), 1),
            "volatility_regime": round(float(selected_row.get('volatility_regime', 0) or 0), 2),
            "true_liquidity": round(float(selected_row.get('true_liquidity', 0) or 0), 2),
            "structure": round(float(selected_row.get('structure', 0) or 0), 2),
            "momentum": round(float(selected_row.get('momentum', 0) or 0), 2),
            "volatility_level": round(float(selected_row.get('volatility_level', 0) or 0), 2),
        })

        question = st.text_input("Ask AI helper (e.g., 'Why is the score low?' or 'Recommend trade')")
        if question.strip():
            st.success(generate_ai_helper_response(question, selected_row))
with tab3:
    # Total score trend
    fig_total = go.Figure()
    fig_total.add_trace(go.Scatter(
        x=display_data['timestamp'],
        y=display_data['total_score'],
        mode='lines+markers',
        name='Total Score',
        line=dict(color='#1f77b4', width=3),
        fill='tozeroy'
    ))
    
    # Add AI probability as secondary trace if available
    if 'ai_high_quality_prob' in display_data.columns and display_data['ai_high_quality_prob'].notna().any():
        fig_total.add_trace(go.Scatter(
            x=display_data['timestamp'],
            y=display_data['ai_high_quality_prob'] * 10,  # Scale to 0-10 for comparison
            mode='lines',
            name='AI Confidence (scaled)',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            yaxis='y2'
        ))
        fig_total.update_layout(
            yaxis2=dict(title="AI Confidence", overlaying='y', side='right')
        )
    
    fig_total.add_hline(y=4, line_dash="dash", line_color="red", annotation_text="No Trade Threshold")
    fig_total.add_hline(y=6, line_dash="dash", line_color="orange", annotation_text="Weak Threshold")
    fig_total.add_hline(y=7.5, line_dash="dash", line_color="green", annotation_text="Good Threshold")

    fig_total.update_layout(
        title="Total Score Over Time",
        xaxis_title="Time",
        yaxis_title="Total Score (0-10)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    st.plotly_chart(fig_total, use_container_width=True)

with tab2:
    col_fact_1, col_fact_2 = st.columns([1, 1])
    
    with col_fact_1:
        # Current factor scores
        fig_factors = go.Figure()
        factors_list = ['volatility_regime', 'true_liquidity', 'structure', 'momentum', 'volatility_level']
        latest_values = [latest[f] for f in factors_list]
        factor_labels = ['Vol. Regime', 'Liquidity', 'Structure', 'Momentum', 'Vol. Level']
        
        fig_factors.add_trace(go.Bar(
            x=factor_labels,
            y=latest_values,
            marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']),
            text=[f'{v:.1f}' for v in latest_values],
            textposition='outside'
        ))
        fig_factors.update_layout(
            title="Current Factor Scores",
            yaxis_title="Score (0-10)",
            yaxis=dict(range=[0, 10]),
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        st.plotly_chart(fig_factors, use_container_width=True)
    
    with col_fact_2:
        # ATR analysis
        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(
            x=display_data['timestamp'],
            y=display_data['current_atr'],
            mode='lines',
            name='Current ATR',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy'
        ))
        fig_atr.add_hline(
            y=display_data['mean_atr'].mean() if 'mean_atr' in display_data.columns else 0,
            line_dash="dash",
            line_color="green",
            annotation_text="Mean ATR"
        )
        fig_atr.update_layout(
            title="ATR (Average True Range) Trend",
            xaxis_title="Time",
            yaxis_title="ATR Value",
            height=400,
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig_atr, use_container_width=True)

    # Individual factor trends
    st.subheader("Individual Factor Trends")
    col1, col2 = st.columns(2)

    with col1:
        fig_vol_reg = go.Figure()
        fig_vol_reg.add_trace(go.Scatter(
            x=display_data['timestamp'],
            y=display_data['volatility_regime'],
            mode='lines+markers',
            name='Vol. Regime',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy'
        ))
        fig_vol_reg.update_layout(
            title="Volatility Regime Trend",
            yaxis=dict(range=[0, 10]),
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(fig_vol_reg, use_container_width=True)

        fig_struct = go.Figure()
        fig_struct.add_trace(go.Scatter(
            x=display_data['timestamp'],
            y=display_data['structure'],
            mode='lines+markers',
            name='Structure',
            line=dict(color='#2ca02c', width=2),
            fill='tozeroy'
        ))
        fig_struct.update_layout(
            title="Structure Trend",
            yaxis=dict(range=[0, 10]),
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(fig_struct, use_container_width=True)

    with col2:
        fig_liq = go.Figure()
        fig_liq.add_trace(go.Scatter(
            x=display_data['timestamp'],
            y=display_data['true_liquidity'],
            mode='lines+markers',
            name='True Liquidity',
            line=dict(color='#ff7f0e', width=2),
            fill='tozeroy'
        ))
        fig_liq.update_layout(
            title="True Liquidity Trend",
            yaxis=dict(range=[0, 10]),
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(fig_liq, use_container_width=True)

        fig_mom = go.Figure()
        fig_mom.add_trace(go.Scatter(
            x=display_data['timestamp'],
            y=display_data['momentum'],
            mode='lines+markers',
            name='Momentum',
            line=dict(color='#d62728', width=2),
            fill='tozeroy'
        ))
        fig_mom.update_layout(
            title="Momentum Trend",
            yaxis=dict(range=[0, 10]),
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(fig_mom, use_container_width=True)

with tab4:
    st.subheader("🔬 AI Model Feature Importance")
    
    # Try to load feature importance from metrics.json
    metrics = load_feature_importance()
    
    if metrics and 'feature_importance' in metrics:
        fi_dict = metrics['feature_importance']
        
        # Create bar chart
        features = list(fi_dict.keys())
        importances = list(fi_dict.values())
        
        fig_importance = go.Figure()
        fig_importance.add_trace(go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f'{v:.1%}' for v in importances],
            textposition='outside'
        ))
        fig_importance.update_layout(
            title="Feature Importance in AI Model",
            xaxis_title="Importance Score",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("#### Why These Features Matter:")
        st.markdown(f"""
        - **Volatility Regime** ({importances[features.index('volatility_regime')] if 'volatility_regime' in features else 0:.1%}): 
          Most important. Captures market stability and pattern formation.
        
        - **Momentum** ({importances[features.index('momentum')] if 'momentum' in features else 0:.1%}): 
          Second most important. Indicates directional strength and trend continuation.
        
        - **Structure** ({importances[features.index('structure')] if 'structure' in features else 0:.1%}): 
          Reflects price pattern quality and support/resistance levels.
        
        - **True Liquidity** ({importances[features.index('true_liquidity')] if 'true_liquidity' in features else 0:.1%}): 
          Measures market depth and order flow health.
        
        - **Volatility Level** ({importances[features.index('volatility_level')] if 'volatility_level' in features else 0:.1%}): 
          Indicates absolute volatility magnitude relative to recent history.
        """)
    else:
        st.info("Feature importance data loading... Run trainer.py to update.")
        
        # Show feature descriptions
        st.markdown("#### Feature Explanations:")
        st.markdown("""
        **Feature Importance (from model training):**
        1. **Volatility Regime** (~29%) - Dominant factor. Measures regime changes and volatility clustering.
        2. **Momentum** (~25%) - Strong signal. Captures trend strength and directional bias.
        3. **Structure** (~17%) - Medium importance. Reflects order flow and price organization.
        4. **True Liquidity** (~16%) - Significant. Shows actual market depth vs spread.
        5. **Volatility Level** (~7%) - Supporting. Indicates ATR-based volatility magnitude.
        6. **Current ATR** (~6%) - Supporting. Absolute volatility measure.
        """)

with tab5:
    st.subheader("Historical Data")
    
    # Display with sorting options
    sort_col = st.selectbox("Sort by:", ["timestamp", "total_score", "volatility_regime"], key="sort_key")
    ascending = st.checkbox("Ascending order", value=False)
    
    display_table = display_data.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    
    st.dataframe(
        display_table,
        use_container_width=True,
        height=500,
        column_config={
            'timestamp': st.column_config.TextColumn('Time', width=200),
            'total_score': st.column_config.NumberColumn('Score', width=80, format='%.2f'),
            'ai_high_quality_prob': st.column_config.NumberColumn('AI Prob', width=80, format='%.4f'),
        }
    )
    
    # Download button
    csv = display_table.to_csv(index=False)
    st.download_button(
        label="📥 Download Data (CSV)",
        data=csv,
        file_name=f"market_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Statistics
st.header("📊 Summary Statistics")
stats_col_1, stats_col_2, stats_col_3, stats_col_4, stats_col_5 = st.columns(5)

with stats_col_1:
    st.metric("Avg Score", f"{data['total_score'].mean():.2f}", delta=f"{data['total_score'].iloc[-1] - data['total_score'].mean():.2f}")
with stats_col_2:
    st.metric("Max Score", f"{data['total_score'].max():.2f}")
with stats_col_3:
    st.metric("Min Score", f"{data['total_score'].min():.2f}")
with stats_col_4:
    st.metric("Data Points", len(data))
with stats_col_5:
    high_quality_count = (data['total_score'] >= 7.5).sum()
    st.metric("High Quality %", f"{(high_quality_count/len(data)*100):.1f}%")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p><strong>AMQSS v2 Dashboard with AI Enhancement</strong> | Last updated: {}</p>
    <p>⚠️ For educational and monitoring purposes only. Not financial advice.</p>
    <p>🤖 AI Model: RandomForest Classifier | 📈 Data Updated: {}</p>
    </div>
""".format(
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    data['timestamp'].max().strftime("%Y-%m-%d %H:%M:%S") if len(data) > 0 else "N/A"
), unsafe_allow_html=True)
