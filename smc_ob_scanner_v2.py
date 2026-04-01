import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
from pathlib import Path

# ── Simple Password Protection ──
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        pwd = st.text_input("Enter Password", type="password")
        if st.button("Login"):
            if pwd == st.secrets["APP_PASSWORD"]:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()

check_password()

st.set_page_config(
    page_title="SMC OB Scanner v2 | NSE",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

HISTORY_DIR = Path("scan_history")
HISTORY_DIR.mkdir(exist_ok=True)

# ─────────────────────── INDICATOR FUNCTIONS ───────────────────────

def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ─────────────────────── NIFTY RS CACHE ───────────────────────

@st.cache_data(ttl=3600)
def get_index_returns():
    results = {}
    for name, ticker in [("nifty50", "^NSEI"), ("nifty500", "^CRSLDX")]:
        try:
            df = yf.download(ticker, period="1y", interval="1wk",
                             progress=False, auto_adjust=True)
            if df is not None and len(df) > 13:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                close = df["Close"].dropna()
                results[f"{name}_1m"] = float((close.iloc[-1] - close.iloc[-5])  / close.iloc[-5]  * 100)
                results[f"{name}_3m"] = float((close.iloc[-1] - close.iloc[-13]) / close.iloc[-13] * 100)
                results[f"{name}_6m"] = float((close.iloc[-1] - close.iloc[-26]) / close.iloc[-26] * 100)
            else:
                results[f"{name}_1m"] = results[f"{name}_3m"] = results[f"{name}_6m"] = 0.0
        except Exception:
            results[f"{name}_1m"] = results[f"{name}_3m"] = results[f"{name}_6m"] = 0.0
    return results

# ─────────────────────── CORE SCANNING LOGIC ───────────────────────

def scan_stock(symbol, cfg, index_returns):
    try:
        ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"

        # ── PRE-FILTER: Market Cap & Daily Volume ──
        if cfg["filter_mcap"] or cfg["filter_vol_min"]:
            try:
                fi = yf.Ticker(ticker).fast_info
                if cfg["filter_mcap"]:
                    mcap = getattr(fi, "market_cap", None)
                    if mcap is None or mcap < cfg["min_mcap_cr"] * 1e7:
                        return None
                if cfg["filter_vol_min"]:
                    avg_vol = getattr(fi, "three_month_average_volume", None)
                    if avg_vol is None or avg_vol < cfg["min_daily_vol"]:
                        return None
            except Exception:
                pass

        # ── Download OHLCV only for stocks that passed pre-filter ──
        period_map = {"1wk": "5y", "1d": "2y"}
        df = yf.download(ticker, period=period_map[cfg["tf"]], interval=cfg["tf"],
                         progress=False, auto_adjust=True)

        if df is None or len(df) < 60:
            return None

        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.dropna().copy()

        df["EMA20"]    = df["Close"].ewm(span=20).mean()
        df["EMA50"]    = df["Close"].ewm(span=50).mean()
        df["RSI"]      = calc_rsi(df["Close"])
        df["OBV"]      = calc_obv(df["Close"], df["Volume"])
        df["ATR"]      = calc_atr(df["High"], df["Low"], df["Close"])
        df["VolAvg20"] = df["Volume"].rolling(20).mean()
        df["RVOL"]     = df["Volume"] / df["VolAvg20"]

        latest  = df.iloc[-1]
        price   = float(latest["Close"])
        score   = 0
        signals = []

        # ── 1. Trend: Price above EMA50 ──
        if cfg["ema_filter"]:
            if price > float(latest["EMA50"]):
                score += 20
                signals.append("Above EMA50")
            else:
                return None

        # ── 2. RSI Filter ──
        rsi_val = float(latest["RSI"])
        if cfg["rsi_filter"] and rsi_val < cfg["rsi_min"]:
            return None
        if rsi_val >= cfg["rsi_min"]:
            score += 10
            signals.append(f"RSI {rsi_val:.1f}")

        # ── 3. Low-Volume Correction Detection ──
        window      = min(15, len(df) - 2)
        low_vol_run = 0
        for i in range(len(df) - 2, len(df) - 2 - window, -1):
            v_avg = float(df["VolAvg20"].iloc[i])
            vol   = float(df["Volume"].iloc[i])
            if v_avg == 0 or np.isnan(v_avg):
                continue
            if (vol / v_avg) < cfg["vol_contract_pct"]:
                low_vol_run += 1
            else:
                break

        correction_ok = low_vol_run >= cfg["min_corr_candles"]
        if correction_ok:
            score += 25
            signals.append(f"Low-Vol Correction ({low_vol_run}c)")

        # ── 4. Bullish Order Block Detection ──
        ob_found = ob_high = ob_low = ob_candle_idx = None
        ob_rvol  = 0.0

        for i in range(len(df) - 2, max(len(df) - 25, 5), -1):
            v_avg = float(df["VolAvg20"].iloc[i])
            if v_avg == 0 or np.isnan(v_avg):
                continue
            rvol_i = float(df["Volume"].iloc[i]) / v_avg
            if rvol_i >= cfg["vol_ob_mult"]:
                fwd = df.iloc[i + 1: i + 4]
                if len(fwd) > 0 and float(fwd["Close"].iloc[-1]) > float(df["Close"].iloc[i]):
                    ob_found      = True
                    ob_high       = float(df["High"].iloc[i])
                    ob_low        = float(df["Low"].iloc[i])
                    ob_rvol       = rvol_i
                    ob_candle_idx = i
                    break

        if ob_found:
            score += 30
            signals.append(f"Bullish OB (RVOL {ob_rvol:.1f}x)")

        # ── 5. Current RVOL ──
        cur_rvol = float(latest["RVOL"]) if not np.isnan(float(latest["RVOL"])) else 0.0
        if cur_rvol >= 2.0:
            score += 15
            signals.append(f"Current RVOL {cur_rvol:.1f}x")
        elif cur_rvol >= 1.5:
            score += 8
            signals.append(f"RVOL {cur_rvol:.1f}x")

        # ── 6. Break of Structure (BOS) ──
        swing_hi_10   = float(df["High"].iloc[-11:-1].max()) if len(df) > 11 else float(df["High"].max())
        bos_confirmed = price > swing_hi_10
        if bos_confirmed:
            score += 15
            signals.append("BOS Confirmed")
        elif cfg["bos_required"]:
            return None

        # ── 7. OBV Divergence ──
        if len(df) > 10:
            price_chg = (float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-6])) / float(df["Close"].iloc[-6])
            obv_base  = float(df["OBV"].iloc[-6])
            obv_chg   = (float(df["OBV"].iloc[-1]) - obv_base) / abs(obv_base) if obv_base != 0 else 0
            if price_chg < 0 and obv_chg > 0:
                score += 10
                signals.append("OBV Divergence")

        # ── 8. Price near OB Zone ──
        if ob_found and ob_low and ob_high:
            ob_range = ob_high - ob_low
            if ob_low <= price <= ob_high + ob_range * 0.5:
                score += 10
                signals.append("Price near OB zone")

        if score < cfg["min_score"]:
            return None

        # ── 52-Week Range ──
        yearly    = df.tail(52) if cfg["tf"] == "1wk" else df.tail(252)
        high_52w  = float(yearly["High"].max())
        low_52w   = float(yearly["Low"].min())
        range_52w = high_52w - low_52w
        pct_in_range = round((price - low_52w) / range_52w * 100, 1) if range_52w > 0 else 0.0

        if cfg["filter_52w"] and pct_in_range > cfg["max_52w_pct"]:
            return None

        # ── All-Time High & Distance ──
        ath      = float(df["High"].max())
        dist_ath = round((price - ath) / ath * 100, 1)

        # ── Relative Strength vs Nifty 50 & 500 ──
        stock_1m = stock_3m = stock_6m = 0.0
        try:
            close = df["Close"].dropna()
            if len(close) >= 26:
                stock_1m = float((close.iloc[-1] - close.iloc[-5])  / close.iloc[-5]  * 100)
                stock_3m = float((close.iloc[-1] - close.iloc[-13]) / close.iloc[-13] * 100)
                stock_6m = float((close.iloc[-1] - close.iloc[-26]) / close.iloc[-26] * 100)
        except Exception:
            pass

        n50_3m  = index_returns.get("nifty50_3m",  1.0)
        n500_3m = index_returns.get("nifty500_3m", 1.0)
        rs_n50  = round(stock_3m / n50_3m,  2) if n50_3m  != 0 else 0.0
        rs_n500 = round(stock_3m / n500_3m, 2) if n500_3m != 0 else 0.0

        if rs_n50 > 1.5:
            score += 10
            signals.append(f"Strong RS vs N50 ({rs_n50}x)")
        elif rs_n50 > 1.0:
            score += 5
            signals.append(f"RS vs N50 ({rs_n50}x)")

        atr_val = float(latest["ATR"]) if not np.isnan(float(latest["ATR"])) else 0

        return {
            "Symbol":        symbol,
            "Price (Rs)":    round(price, 2),
            "Score":         score,
            "RVOL":          round(cur_rvol, 2),
            "RSI":           round(rsi_val, 1),
            "Low Vol Corr":  "YES" if correction_ok else "NO",
            "Bullish OB":    "YES" if ob_found      else "NO",
            "BOS":           "YES" if bos_confirmed  else "NO",
            "OB High":       round(ob_high, 2) if ob_high else 0,
            "OB Low":        round(ob_low,  2) if ob_low  else 0,
            "SL Suggest":    round(price - 1.5 * atr_val, 2),
            "52W Range Pct": pct_in_range,
            "52W High":      round(high_52w, 2),
            "52W Low":       round(low_52w,  2),
            "ATH":           round(ath, 2),
            "Dist ATH Pct":  dist_ath,
            "Ret 1M Pct":    round(stock_1m, 1),
            "Ret 3M Pct":    round(stock_3m, 1),
            "Ret 6M Pct":    round(stock_6m, 1),
            "RS vs N50":     rs_n50,
            "RS vs N500":    rs_n500,
            "Signals":       " | ".join(signals),
            "_df":           df,
            "_ob_idx":       ob_candle_idx,
        }

    except Exception:
        return None

# ─────────────────────── CHART RENDERER ───────────────────────

def render_chart(symbol, df, ob_idx, timeframe):
    df  = df.tail(80).copy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.55, 0.25, 0.20],
                        subplot_titles=["", "Volume", "RSI"])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20",
                             line=dict(color="#f39c12", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50",
                             line=dict(color="#3498db", width=1.5)), row=1, col=1)

    if ob_idx is not None and ob_idx < len(df):
        ob = df.iloc[ob_idx]
        fig.add_hrect(y0=float(ob["Low"]), y1=float(ob["High"]),
                      fillcolor="rgba(52,152,219,0.15)",
                      line=dict(color="#3498db", width=1, dash="dot"),
                      annotation_text="Bullish OB",
                      annotation_position="top right", row=1, col=1)

    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                         marker_color=colors, name="Volume", opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["VolAvg20"], name="Vol Avg 20",
                             line=dict(color="#e74c3c", width=1.5, dash="dash")), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#9b59b6", width=1.5)), row=3, col=1)
    for level, col in [(70, "#ef5350"), (50, "#ffffff"), (30, "#26a69a")]:
        fig.add_hline(y=level, line=dict(color=col, width=0.8, dash="dot"), row=3, col=1)

    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> - {'Weekly' if timeframe == '1wk' else 'Daily'} Chart",
            font=dict(size=18)
        ),
        template="plotly_dark", height=680,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    )
    return fig

# ─────────────────────── SCAN HISTORY ───────────────────────

def save_scan(df_results, label=""):
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{ts}_{label}" if label else ts
    path = HISTORY_DIR / f"scan_{name}.csv"
    df_results.drop(columns=["Signals"], errors="ignore").to_csv(path, index=False)
    return path

def load_scan_history():
    return sorted(HISTORY_DIR.glob("scan_*.csv"), reverse=True)

def compare_scans(df_new, df_old):
    new_syms = set(df_new["Symbol"])
    old_syms = set(df_old["Symbol"])
    appeared = new_syms - old_syms
    dropped  = old_syms - new_syms
    common   = new_syms & old_syms

    score_changes = []
    for sym in common:
        s_new = df_new[df_new["Symbol"] == sym]["Score"].values[0]
        s_old = df_old[df_old["Symbol"] == sym]["Score"].values[0]
        diff  = s_new - s_old
        if diff != 0:
            score_changes.append({"Symbol": sym, "Old Score": s_old,
                                  "New Score": s_new, "Change": diff})

    sc_df = pd.DataFrame(score_changes).sort_values("Change", ascending=False) \
            if score_changes else pd.DataFrame()
    return appeared, dropped, sc_df

def to_tradingview_list(symbols):
    return ",".join([f"NSE:{s}" for s in symbols])

# ─────────────────────── SIDEBAR ───────────────────────

with st.sidebar:
    st.title("Scanner Settings")

    st.subheader("Timeframe")
    timeframe = st.selectbox("", ["1wk", "1d"], index=0,
                             format_func=lambda x: "Weekly" if x == "1wk" else "Daily")

    st.subheader("Volume Filters")
    vol_contract_pct = st.slider("Correction Vol % of Avg", 20, 70, 50) / 100
    vol_ob_mult      = st.slider("OB Vol Min Multiplier (x)", 1.5, 4.0, 2.0, step=0.1)
    min_corr_candles = st.slider("Min Low-Vol Candles", 2, 8, 3)

    st.subheader("Structure Filters")
    ema_filter   = st.checkbox("Price > EMA 50", value=True)
    bos_required = st.checkbox("Require BOS", value=False)

    st.subheader("RSI Filter")
    rsi_filter = st.checkbox("Enable RSI Filter", value=True)
    rsi_min    = st.slider("RSI Minimum", 20, 70, 30)

    st.subheader("Market Cap Filter")
    filter_mcap = st.checkbox("Enable Market Cap Filter", value=True)
    min_mcap_cr = st.slider("Min Market Cap (Rs Cr)", 100, 5000, 500, step=100)

    st.subheader("Daily Volume Filter")
    filter_vol_min = st.checkbox("Enable Min Daily Volume", value=True)
    min_daily_vol  = st.slider("Min Avg Daily Volume", 50000, 1000000, 200000, step=50000)

    st.subheader("52-Week Range Filter")
    filter_52w  = st.checkbox("Enable 52W Range Filter", value=False)
    max_52w_pct = st.slider("Max 52W Range % (exclude overbought)", 50, 100, 80)
    st.caption("80 = exclude stocks in top 20% of 52W range")

    st.subheader("Score Threshold")
    min_score = st.slider("Minimum Score", 30, 80, 40)

    st.markdown("---")
    st.markdown("**Score Breakdown**")
    for k, v in {
        "EMA50 Above": 20, "RSI Filter": 10, "Low Vol Corr": 25,
        "Bullish OB": 30, "High RVOL": 15, "BOS": 15,
        "OBV Divergence": 10, "Near OB": 10, "Strong RS N50": 10
    }.items():
        st.markdown(f"<small>* {k}  +{v}</small>", unsafe_allow_html=True)
    st.markdown("<small>**Max Score: 145**</small>", unsafe_allow_html=True)

cfg = {
    "tf":               timeframe,
    "vol_contract_pct": vol_contract_pct,
    "vol_ob_mult":      vol_ob_mult,
    "min_corr_candles": min_corr_candles,
    "ema_filter":       ema_filter,
    "rsi_filter":       rsi_filter,
    "rsi_min":          rsi_min,
    "bos_required":     bos_required,
    "filter_52w":       filter_52w,
    "max_52w_pct":      max_52w_pct,
    "min_score":        min_score,
    "filter_mcap":      filter_mcap,
    "min_mcap_cr":      min_mcap_cr,
    "filter_vol_min":   filter_vol_min,
    "min_daily_vol":    min_daily_vol,
}

# ─────────────────────── MAIN UI ───────────────────────

st.title("SMC Bullish OB + Volume Scanner v2")
st.caption("Low Volume Correction | Bullish OB | BOS | RS vs Nifty | 52W Range | ATH Distance | Scan History")

tab_scan, tab_history, tab_how = st.tabs(["Scanner", "Scan History", "How It Works"])

# ══════════════ TAB 1: SCANNER ══════════════
with tab_scan:
    uploaded_file = st.file_uploader("Upload NSE Stock List (.txt)", type=["txt"])

    if uploaded_file:
        raw     = uploaded_file.read().decode("utf-8")
        symbols = [l.strip().upper() for l in raw.splitlines() if l.strip()]
        symbols = [s.replace(".NS", "").replace(".NSE", "") for s in symbols]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Stocks Loaded",  len(symbols))
        c2.metric("Timeframe",      "Weekly" if timeframe == "1wk" else "Daily")
        c3.metric("Min Score",      f"{min_score}/145")
        c4.metric("Min Mkt Cap",    f"Rs {min_mcap_cr} Cr" if filter_mcap else "OFF")
        c5.metric("Min Daily Vol",  f"{min_daily_vol:,}" if filter_vol_min else "OFF")

        st.markdown("---")
        scan_label = st.text_input("Scan Label (optional)", placeholder="e.g. Weekly_Swing_Mar2026")

        if st.button("Run Scanner", use_container_width=True, type="primary"):

            with st.spinner("Fetching Nifty 50 & Nifty 500 data..."):
                index_returns = get_index_returns()

            st.success(
                f"Nifty50 3M: {index_returns['nifty50_3m']:.1f}%  |  "
                f"Nifty500 3M: {index_returns['nifty500_3m']:.1f}%"
            )

            results    = []
            prog_bar   = st.progress(0)
            status_txt = st.empty()

            for idx, sym in enumerate(symbols):
                status_txt.markdown(f"Scanning **{sym}** ({idx+1}/{len(symbols)})...")
                res = scan_stock(sym, cfg, index_returns)
                if res:
                    results.append(res)
                prog_bar.progress((idx + 1) / len(symbols))
                if (idx + 1) % 5 == 0:
                    time.sleep(0.3)

            prog_bar.empty()
            status_txt.success(
                f"Scan Complete - {len(results)} setups found from {len(symbols)} stocks"
            )

            if results:
                df_clean = pd.DataFrame([
                    {k: v for k, v in r.items() if not k.startswith("_")}
                    for r in results
                ])
                df_clean = df_clean.sort_values("Score", ascending=False).reset_index(drop=True)

                saved_path = save_scan(df_clean, scan_label)
                st.info(f"Scan saved: {saved_path.name}")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Matches",  len(df_clean))
                m2.metric("With OB",        len(df_clean[df_clean["Bullish OB"] == "YES"]))
                m3.metric("Low Vol Corr",   len(df_clean[df_clean["Low Vol Corr"] == "YES"]))
                m4.metric("BOS Confirmed",  len(df_clean[df_clean["BOS"] == "YES"]))
                m5.metric("RS > 1 (N50)",   len(df_clean[df_clean["RS vs N50"] > 1.0]))

                st.markdown("### Scan Results")

                display_cols = [
                    "Symbol", "Price (Rs)", "Score", "RVOL", "RSI",
                    "Low Vol Corr", "Bullish OB", "BOS",
                    "52W Range Pct", "Dist ATH Pct",
                    "RS vs N50", "RS vs N500",
                    "Ret 1M Pct", "Ret 3M Pct",
                    "OB High", "OB Low", "SL Suggest"
                ]

                                def color_score(val):
                    if val >= 90:
                        return "color: #26a69a; font-weight: bold"
                    elif val >= 60:
                        return "color: #f39c12; font-weight: bold"
                    else:
                        return "color: #ef5350"

                def color_rs(val):
                    if val >= 1.5:
                        return "color: #26a69a; font-weight: bold"
                    elif val >= 1.0:
                        return "color: #f39c12"
                    else:
                        return "color: #ef5350"

                def color_dist_ath(val):
                    if val >= -10:
                        return "color: #ef5350"
                    elif val >= -30:
                        return "color: #f39c12"
                    else:
                        return "color: #26a69a; font-weight: bold"

                styled = (
                    df_clean[display_cols]
                    .style
                    .map(color_score,    subset=["Score"])
                    .map(color_rs,       subset=["RS vs N50", "RS vs N500"])
                    .map(color_dist_ath, subset=["Dist ATH Pct"])
                    .format({
                        "Price (Rs)":    "Rs {:.2f}",
                        "RVOL":          "{:.2f}x",
                        "RSI":           "{:.1f}",
                        "SL Suggest":    "Rs {:.2f}",
                        "52W Range Pct": "{:.1f}%",
                        "Dist ATH Pct":  "{:.1f}%",
                        "RS vs N50":     "{:.2f}x",
                        "RS vs N500":    "{:.2f}x",
                        "Ret 1M Pct":    "{:.1f}%",
                        "Ret 3M Pct":    "{:.1f}%",
                        "OB High":       "Rs {:.2f}",
                        "OB Low":        "Rs {:.2f}",
                    })
                )
                st.dataframe(styled, use_container_width=True, height=450)

                st.markdown("### Export")
                dcol1, dcol2, dcol3 = st.columns(3)

                csv = df_clean.drop(columns=["Signals"], errors="ignore").to_csv(index=False)
                dcol1.download_button("Download CSV", data=csv,
                    file_name=f"smc_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv", use_container_width=True)

                tv_all = to_tradingview_list(df_clean["Symbol"].tolist())
                dcol2.download_button("TradingView Watchlist (All)", data=tv_all,
                    file_name=f"tv_watchlist_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain", use_container_width=True)

                tv_top = to_tradingview_list(df_clean.head(20)["Symbol"].tolist())
                dcol3.download_button("TradingView Top 20", data=tv_top,
                    file_name=f"tv_top20_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain", use_container_width=True)

                st.caption("TradingView Import: Watchlist > 3 dots > Import List > paste file content")

                st.markdown("### Relative Strength Leaderboard (Top 15 vs Nifty 50)")
                rs_top = df_clean.nlargest(15, "RS vs N50")[[
                    "Symbol", "Price (Rs)", "RS vs N50", "RS vs N500",
                    "Ret 1M Pct", "Ret 3M Pct", "Score"
                ]]
                st.dataframe(rs_top, use_container_width=True, hide_index=True)

                st.markdown("### 52-Week Range Distribution")
                bins   = [0, 25, 50, 75, 100]
                labels = ["Bottom 25%", "25-50%", "50-75%", "Top 25%"]
                df_clean["52W Bucket"] = pd.cut(df_clean["52W Range Pct"],
                                                bins=bins, labels=labels)
                bucket_counts = df_clean["52W Bucket"].value_counts().sort_index()
                fig_dist = go.Figure(go.Bar(
                    x=bucket_counts.index.tolist(),
                    y=bucket_counts.values.tolist(),
                    marker_color=["#26a69a", "#2ecc71", "#f39c12", "#ef5350"],
                    text=bucket_counts.values.tolist(), textposition="auto"
                ))
                fig_dist.update_layout(template="plotly_dark", height=300,
                                       title="Stocks by 52-Week Range Position",
                                       paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
                st.plotly_chart(fig_dist, use_container_width=True)

                st.markdown("---")
                st.markdown("### Chart Viewer")
                sel_sym = st.selectbox("Select stock:", df_clean["Symbol"].tolist())

                if sel_sym:
                    row = next(r for r in results if r["Symbol"] == sel_sym)
                    st.plotly_chart(
                        render_chart(sel_sym, row["_df"], row["_ob_idx"], timeframe),
                        use_container_width=True
                    )

                    st.markdown("**Detected Signals:**")
                    scols = st.columns(4)
                    for i, sig in enumerate(row["Signals"].split(" | ")):
                        scols[i % 4].success(sig)

                    sel_row = df_clean[df_clean["Symbol"] == sel_sym].iloc[0]
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("RS vs Nifty 50",  f"{sel_row['RS vs N50']:.2f}x",
                               delta="Outperforming" if sel_row["RS vs N50"] > 1 else "Underperforming")
                    rc2.metric("RS vs Nifty 500", f"{sel_row['RS vs N500']:.2f}x",
                               delta="Outperforming" if sel_row["RS vs N500"] > 1 else "Underperforming")
                    rc3.metric("52W Range",        f"{sel_row['52W Range Pct']:.1f}%")
                    rc4.metric("Dist from ATH",    f"{sel_row['Dist ATH Pct']:.1f}%")

                    ob_high = row["OB High"]
                    ob_low  = row["OB Low"]
                    if ob_high and ob_low and ob_high != 0 and ob_low != 0:
                        st.markdown("---")
                        st.markdown("### Tranche Entry Plan")
                        t1 = round(float(ob_low) + (float(ob_high) - float(ob_low)) * 0.3, 2)
                        t2 = round(float(ob_high) * 1.01, 2)
                        t3 = round(sel_row["Price (Rs)"] * 1.05, 2)
                        tc1, tc2, tc3 = st.columns(3)
                        tc1.info(f"**T1 - 50% allocation**\n\nOB Retest Zone\nRs {ob_low} to Rs {ob_high}\nEntry ~Rs {t1}")
                        tc2.info(f"**T2 - 30% allocation**\n\nPost BOS Confirmation\nEntry ~Rs {t2}")
                        tc3.info(f"**T3 - 20% allocation**\n\nMomentum Continuation\nEntry ~Rs {t3}")
                        st.warning(
                            f"Stop Loss: Rs {row['SL Suggest']} (1.5x ATR) | "
                            f"Move to Break-Even before adding T2"
                        )
            else:
                st.warning("No stocks matched. Try relaxing the sidebar filters.")
    else:
        st.info("Upload your NSE stock list (.txt) to begin scanning")
        st.code("RELIANCE\nTCS\nINFY\nCUPID\nHDFCBANK", language="text")

# ══════════════ TAB 2: SCAN HISTORY ══════════════
with tab_history:
    st.markdown("### Saved Scan History")
    history_files = load_scan_history()

    if not history_files:
        st.info("No scan history yet. Run a scan first.")
    else:
        scan_names = [f.name for f in history_files]
        st.success(f"Found **{len(scan_names)}** saved scans")

        st.markdown("#### View a Saved Scan")
        sel_scan = st.selectbox("Select scan:", scan_names, key="view_scan")
        if sel_scan:
            df_hist = pd.read_csv(HISTORY_DIR / sel_scan)
            st.dataframe(df_hist, use_container_width=True, height=350)
            col_dl1, col_dl2 = st.columns(2)
            col_dl1.download_button("Download CSV",
                data=df_hist.to_csv(index=False),
                file_name=sel_scan, mime="text/csv", use_container_width=True)
            if "Symbol" in df_hist.columns:
                col_dl2.download_button("Export to TradingView",
                    data=to_tradingview_list(df_hist["Symbol"].tolist()),
                    file_name=f"tv_{sel_scan.replace('.csv', '.txt')}",
                    mime="text/plain", use_container_width=True)

        st.markdown("---")
        st.markdown("#### Compare Two Scans")

        if len(scan_names) >= 2:
            col_a, col_b = st.columns(2)
            scan_a = col_a.selectbox("Newer Scan:", scan_names, index=0, key="scan_a")
            scan_b = col_b.selectbox("Older Scan:", scan_names, index=1, key="scan_b")

            if st.button("Compare Scans", use_container_width=True):
                df_a = pd.read_csv(HISTORY_DIR / scan_a)
                df_b = pd.read_csv(HISTORY_DIR / scan_b)
                appeared, dropped, score_df = compare_scans(df_a, df_b)

                ca, cd, cs = st.columns(3)
                ca.metric("New Stocks",    len(appeared))
                cd.metric("Dropped",       len(dropped))
                cs.metric("Score Changed", len(score_df))

                tab_new, tab_drop, tab_chg = st.tabs([
                    "Newly Appeared", "Dropped", "Score Changes"
                ])

                with tab_new:
                    if appeared:
                        df_new_stocks = df_a[df_a["Symbol"].isin(appeared)].sort_values(
                            "Score", ascending=False)
                        st.dataframe(df_new_stocks, use_container_width=True, hide_index=True)
                        st.download_button("Export New Stocks to TradingView",
                            data=to_tradingview_list(df_new_stocks["Symbol"].tolist()),
                            file_name="tv_new_stocks.txt", mime="text/plain")
                    else:
                        st.info("No new stocks in the newer scan.")

                with tab_drop:
                    if dropped:
                        df_drop = df_b[df_b["Symbol"].isin(dropped)].sort_values(
                            "Score", ascending=False)
                        st.dataframe(df_drop, use_container_width=True, hide_index=True)
                    else:
                        st.info("No stocks dropped from previous scan.")

                with tab_chg:
                    if not score_df.empty:
                        def color_change(val):
                            return "color: #26a69a; font-weight:bold" if val > 0 else "color: #ef5350"
                        st.dataframe(
                            score_df.style.applymap(color_change, subset=["Change"]),
                            use_container_width=True, hide_index=True)
                    else:
                        st.info("No score changes detected.")
        else:
            st.info("Run at least 2 scans to enable comparison.")

# ══════════════ TAB 3: HOW IT WORKS ══════════════
with tab_how:
    st.markdown("""
    ## SMC OB Scanner v2 - Complete Feature Guide

    ### Pattern Being Detected
    > Low Volume Correction → Bullish Order Block (High Volume) → Price Explosion

    Mirrors the exact setup in **Cupid Limited (NSE)** — accumulation on very low volumes,
    followed by institutional high-volume OB candle, then multi-bagger move.

    ---

    ### All Filters

    | Filter | Default | Purpose |
    |---|---|---|
    | Price > EMA50 | ON | Stock must be in uptrend |
    | RSI Minimum | 30 | Catches early-stage recoveries |
    | Market Cap | Rs 500 Cr+ | Removes micro/penny stocks |
    | Avg Daily Volume | 2,00,000+ | Ensures liquidity |
    | 52W Range % | OFF | Exclude overbought stocks |
    | BOS Required | OFF | Optional hard structure filter |

    ---

    ### Scoring System (Max 145)

    | Condition | Points |
    |---|---|
    | Price > EMA50 | 20 |
    | RSI >= Minimum | 10 |
    | Low Volume Correction | 25 |
    | Bullish OB Detected | 30 |
    | Current RVOL >= 2x | 15 |
    | BOS Confirmed | 15 |
    | OBV Divergence | 10 |
    | Price near OB Zone | 10 |
    | Strong RS vs N50 >1.5x | 10 |

    ---

    ### Best Setup Checklist (Cupid-Like)
    - Score >= 80
    - 52W Range % < 60%
    - Dist ATH % < -30%
    - RS vs N50 > 1.0x
    - Low Vol Corr = YES + Bullish OB = YES + BOS = YES

    ---

    ### TradingView Import Steps
    1. Download .txt file from Export section
    2. TradingView > Watchlist > 3 dots > Import List
    3. Paste file contents > Import

    ---

    ### Tranche Entry Strategy
    - **T1 (50%)** - Enter at OB retest on low volume
    - **T2 (30%)** - Add after BOS confirmation on high volume
    - **T3 (20%)** - Add on momentum with RVOL >= 2x
    - **Stop Loss** - 1.5x ATR. Trail to break-even before each addition
    """)