"""
Personalized Stock Decision Support System
Streamlit UI — calls get_decision() from the ML notebook logic
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Decision Support",
    page_icon="📈",
    layout="centered",
)

# ──────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS — dark bluish professional theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1b2a;
        color: #dce8f5;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] { display: none; }

    /* ── Title ── */
    h1 { color: #4fc3f7; letter-spacing: 1px; }
    h2, h3 { color: #81d4fa; }

    /* ── Input widgets ── */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background: #1a2f45 !important;
        color: #dce8f5 !important;
        border: 1px solid #2e5373 !important;
        border-radius: 8px !important;
    }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #0288d1);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* ── Decision badge ── */
    .decision-badge {
        display: inline-block;
        padding: 12px 40px;
        border-radius: 12px;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 3px;
        margin: 10px 0;
    }
    .badge-buy  { background: #0a3d1f; color: #00e676; border: 2px solid #00e676; }
    .badge-sell { background: #3d0a0a; color: #ff5252; border: 2px solid #ff5252; }
    .badge-hold { background: #3d2d0a; color: #ffd740; border: 2px solid #ffd740; }

    /* ── Metric cards ── */
    .metric-card {
        background: #1a2f45;
        border: 1px solid #2e5373;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 0.8rem; color: #81d4fa; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #ffffff; margin-top: 4px; }
    .metric-sub   { font-size: 0.85rem; color: #90a4ae; margin-top: 4px; }

    /* ── Explanation box ── */
    .explanation-box {
        background: #132235;
        border-left: 4px solid #0288d1;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 0.95rem;
        color: #cfd8e3;
        margin-top: 10px;
    }

    /* ── Divider ── */
    hr { border-color: #1e3a50; }

    /* ── Matplotlib canvas transparent ── */
    .stImage { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  ML PIPELINE  (mirrors the notebook logic — self-contained)
# ──────────────────────────────────────────────────────────────────────────────

def _fetch(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for '{ticker}'. Check the ticker symbol.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()].reset_index()
    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[needed].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates("Date")
    df = df[df["Volume"] > 0].ffill().reset_index(drop=True)
    return df


def _features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA_5"]  = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    df["Daily_Return"]  = df["Close"].pct_change() * 100
    df["Volatility_10"] = df["Daily_Return"].rolling(10).std()
    df["Volume_Ratio"]  = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["Target_Close"]  = df["Close"].shift(-1)
    return df.dropna().reset_index(drop=True)


FEAT_COLS = [
    "Close", "MA_5", "MA_20", "MA_50",
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "Daily_Return", "Volatility_10", "Volume_Ratio", "Volume",
]


@st.cache_data(show_spinner=False)
def get_decision(stock: str, quantity: int) -> dict:
    """
    Full pipeline: fetch → preprocess → engineer → train → predict → decide.
    Returns a structured dict ready for the UI.
    """
    ticker = stock.strip().upper()
    df = _preprocess(_fetch(ticker))
    df = _features(df)

    X, y = df[FEAT_COLS].values, df["Target_Close"].values
    split = int(len(X) * 0.85)
    X_tr, X_te = X[:split], X[split:]
    y_tr, _    = y[:split], y[split:]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_split=5, random_state=42, n_jobs=-1,
    )
    model.fit(X_tr_s, y_tr)

    cv_r2       = cross_val_score(model, X_tr_s, y_tr, cv=5, scoring="r2").mean()
    confidence  = round(max(0.0, min(1.0, cv_r2)) * 100, 2)

    latest_s    = sc.transform(X[-1].reshape(1, -1))
    pred_price  = float(model.predict(latest_s)[0])
    curr_price  = float(df["Close"].iloc[-1])
    exp_return  = ((pred_price - curr_price) / curr_price) * 100

    # Decision
    if exp_return > 1.0:
        decision = "BUY"
        qty_delta = max(1, round(quantity * 0.20))
        action    = f"Consider buying {qty_delta} additional share(s)."
    elif exp_return < -0.5:
        decision = "SELL"
        qty_delta = max(1, round(quantity * 0.30))
        action    = f"Consider selling {qty_delta} share(s) to reduce exposure."
    else:
        decision = "HOLD"
        qty_delta = 0
        action    = "No action needed. Monitor closely."

    # Risk
    vol = float(df["Volatility_10"].iloc[-1])
    risk = "LOW" if vol < 1.0 else "MEDIUM" if vol < 2.5 else "HIGH"

    # Explanation
    rsi  = float(df["RSI"].iloc[-1])
    macd = float(df["MACD_Hist"].iloc[-1])
    ma20 = float(df["MA_20"].iloc[-1])
    rsi_note  = ("RSI overbought (>70)" if rsi > 70
                 else "RSI oversold (<30)" if rsi < 30
                 else f"RSI neutral ({rsi:.1f})")
    macd_note = ("MACD bullish (positive histogram)" if macd > 0
                 else "MACD bearish (negative histogram)")
    trend_note = "Price above MA_20 (uptrend)" if curr_price > ma20 else "Price below MA_20 (downtrend)"
    explanation = f"{trend_note}. {rsi_note}. {macd_note}. 10-day volatility: {vol:.2f}%."

    return {
        "ticker":              ticker,
        "current_price":       round(curr_price, 4),
        "predicted_price":     round(pred_price, 4),
        "expected_return_pct": round(exp_return, 4),
        "decision":            decision,
        "suggested_action":    action,
        "confidence_score":    confidence,
        "risk_level":          risk,
        "explanation":         explanation,
        "_df":                 df,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  CHART HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor("#0d1b2a")
    ax.tick_params(colors="#90a4ae")
    ax.xaxis.label.set_color("#81d4fa")
    ax.yaxis.label.set_color("#81d4fa")
    ax.title.set_color("#4fc3f7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e5373")
    ax.grid(True, alpha=0.2, color="#2e5373")


def plot_price_ma(df: pd.DataFrame, ticker: str, predicted_price: float):
    tail = df.tail(120)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0d1b2a")
    ax.plot(tail["Date"], tail["Close"],  label="Close",  lw=1.8, color="#4fc3f7")
    ax.plot(tail["Date"], tail["MA_20"],  label="MA 20",  lw=1.3, linestyle="--", color="#ffd740")
    ax.plot(tail["Date"], tail["MA_50"],  label="MA 50",  lw=1.3, linestyle=":",  color="#ff7043")
    ax.scatter([tail["Date"].iloc[-1]], [predicted_price],
               color="#00e676", s=80, zorder=5, label=f"Predicted ${predicted_price:.2f}")
    ax.set_title(f"{ticker} — Price & Moving Averages (120 days)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=9, facecolor="#1a2f45", labelcolor="#dce8f5")
    _style_ax(ax)
    fig.tight_layout()
    return fig


def plot_rsi(df: pd.DataFrame, ticker: str):
    tail = df.tail(120)
    fig, ax = plt.subplots(figsize=(9, 3), facecolor="#0d1b2a")
    ax.plot(tail["Date"], tail["RSI"], lw=1.4, color="#ba68c8", label="RSI (14)")
    ax.axhline(70, color="#ff5252", linestyle="--", lw=1, label="Overbought 70")
    ax.axhline(30, color="#00e676", linestyle="--", lw=1, label="Oversold 30")
    ax.fill_between(tail["Date"], 30, 70, alpha=0.06, color="gray")
    ax.set_ylim(0, 100)
    ax.set_title(f"{ticker} — RSI", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI")
    ax.legend(fontsize=9, facecolor="#1a2f45", labelcolor="#dce8f5")
    _style_ax(ax)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  UI LAYOUT
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("# 📈 Stock Decision Support")
st.markdown("Enter a stock ticker and quantity owned to get an AI-powered BUY / HOLD / SELL recommendation.")
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    ticker_input = st.text_input(
        "Stock Ticker",
        value="AAPL",
        placeholder="e.g. AAPL, TSLA, RELIANCE.NS",
    )
with col2:
    qty_input = st.number_input(
        "Quantity Owned",
        min_value=1,
        max_value=100_000,
        value=50,
        step=1,
    )

run_btn = st.button("🔍 Analyze")

if run_btn:
    with st.spinner(f"Analyzing {ticker_input.strip().upper()} …"):
        try:
            res = get_decision(ticker_input, int(qty_input))
        except Exception as err:
            st.error(f"❌ {err}")
            st.stop()

    st.markdown("---")

    # ── Decision badge ────────────────────────────────────────────────────────
    badge_class = {"BUY": "badge-buy", "SELL": "badge-sell", "HOLD": "badge-hold"}[res["decision"]]
    st.markdown(
        f"<div style='text-align:center'>"
        f"<span class='decision-badge {badge_class}'>{res['decision']}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Metric cards ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    sign = "+" if res["expected_return_pct"] >= 0 else ""
    ret_color = "#00e676" if res["expected_return_pct"] > 0 else "#ff5252" if res["expected_return_pct"] < 0 else "#ffd740"

    m1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">${res['current_price']:.2f}</div>
        </div>""", unsafe_allow_html=True)

    m2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Price</div>
            <div class="metric-value">${res['predicted_price']:.2f}</div>
        </div>""", unsafe_allow_html=True)

    m3.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Expected Return</div>
            <div class="metric-value" style="color:{ret_color}">{sign}{res['expected_return_pct']:.2f}%</div>
        </div>""", unsafe_allow_html=True)

    m4.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{res['confidence_score']:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    # ── Second row: action + risk ─────────────────────────────────────────────
    a1, a2 = st.columns([3, 1])
    risk_color = {"LOW": "#00e676", "MEDIUM": "#ffd740", "HIGH": "#ff5252"}[res["risk_level"]]

    a1.markdown(f"""
        <div class="metric-card" style="text-align:left">
            <div class="metric-label">Suggested Action</div>
            <div class="metric-value" style="font-size:1rem;margin-top:6px">{res['suggested_action']}</div>
        </div>""", unsafe_allow_html=True)

    a2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Level</div>
            <div class="metric-value" style="color:{risk_color}">{res['risk_level']}</div>
        </div>""", unsafe_allow_html=True)

    # ── Explanation ───────────────────────────────────────────────────────────
    st.markdown(f"""
        <div class="explanation-box">
            <strong>📊 Analysis:</strong><br>{res['explanation']}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("### Price & Moving Averages")
    fig1 = plot_price_ma(res["_df"], res["ticker"], res["predicted_price"])
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    st.markdown("### RSI Indicator")
    fig2 = plot_rsi(res["_df"], res["ticker"])
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.markdown("---")
    st.caption("⚠️ This tool is for educational purposes only and does not constitute financial advice.")
