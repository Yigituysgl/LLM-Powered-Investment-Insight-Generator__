
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import timedelta
from dotenv import load_dotenv, find_dotenv


from litellm import completion


_ = load_dotenv(find_dotenv(usecwd=True))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "groq/llama-3.1-8b-instant")

st.set_page_config(page_title="LLM-Powered Investment Insight", layout="wide")


@st.cache_data
def load_data():
    # prefer merged file if present
    if os.path.exists("merged_msft_with_signals.csv"):
        df = pd.read_csv("merged_msft_with_signals.csv", parse_dates=["date"])
    else:
        df = pd.read_csv("features_MSFT.csv", parse_dates=["date"])
    # light safety
    for c in ["PX","ret_1d","rsi_14","macd","bb_pos","avg_sentiment"]:
        if c in df.columns: 
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)

df = load_data()
min_d, max_d = df["date"].min(), df["date"].max()

st.sidebar.header("View")
lookback_days = st.sidebar.slider("Lookback (days)", 30, 365, 120)
end_date = st.sidebar.date_input("End date", value=max_d.date(), min_value=min_d.date(), max_value=max_d.date())
start_date = pd.to_datetime(end_date) - timedelta(days=lookback_days)
mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
view = df.loc[mask].copy()

st.sidebar.header("LLM")
st.sidebar.write("Model:", DEFAULT_MODEL)
if not GROQ_API_KEY:
    st.sidebar.error("GROQ_API_KEY not found. Locally, add it to .env. On Streamlit Cloud, put it in Secrets.")
else:
    st.sidebar.success("API key loaded")


def plot_price_signals_sentiment(data: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax2 = ax1.twinx()

    
    ax1.plot(data["date"], data["PX"], linewidth=2, label="Price")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Price")

    
    if "signal" in data.columns:
        long_idx  = data["signal"].fillna(0) == 1
        short_idx = data["signal"].fillna(0) == -1
        ax1.scatter(data.loc[long_idx, "date"],  data.loc[long_idx, "PX"],  s=40, marker="^", label="Long")
        ax1.scatter(data.loc[short_idx, "date"], data.loc[short_idx, "PX"], s=40, marker="v", label="Short")

    
    if "avg_sentiment" in data.columns:
        scaled = data["avg_sentiment"] * 200.0
        ax2.plot(data["date"], scaled, linestyle="--", alpha=.8, label="Sentiment (×200)")
        ax2.axhline(0, color="gray", lw=1, alpha=.4)
        ax2.set_ylabel("Scaled Sentiment")

    ax1.grid(alpha=.3)
    ax1.set_title("Price, Signals & Sentiment")
    
    lines, labels = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+l2, labels+lb2, loc="upper left")
    fig.tight_layout()
    return fig

def plot_equity_curves(data: pd.DataFrame):
    sig = data.get("signal", pd.Series(0, index=data.index)).fillna(0).shift(1).fillna(0)
    strat_ret = sig * data["ret_1d"].fillna(0)
    bh_ret    = data["ret_1d"].fillna(0)
    strat_curve = (1 + strat_ret).cumprod()
    bh_curve    = (1 + bh_ret).cumprod()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(data["date"], strat_curve, label="Strategy")
    ax.plot(data["date"], bh_curve, label="Buy & Hold", alpha=.8)
    ax.set_title("Strategy vs Buy & Hold")
    ax.set_ylabel("Growth of $1"); ax.set_xlabel("Date")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    return fig


SYS_PROMPT = (
    "You are a professional financial analyst who writes concise, data-driven insights. "
    "Output bullet points and a short conclusion."
)

def row_context(row):
    
    return (
        f"Date: {row['date']:%Y-%m-%d}\n"
        f"Price PX: {row['PX']:.2f}\n"
        f"1D return: {row.get('ret_1d', np.nan):+.3%}\n"
        f"RSI(14): {row.get('rsi_14', np.nan):.1f}\n"
        f"MACD: {row.get('macd', np.nan):.2f}\n"
        f"Bollinger position: {row.get('bb_pos', np.nan):.2f}\n"
        f"Avg news sentiment: {row.get('avg_sentiment', np.nan):.2f}\n"
        f"Strategy signal (1=long,0=flat,-1=short): {int(row.get('signal', 0))}\n"
    )

def llm_insight_from_row(row):
    if not GROQ_API_KEY:
        return "⚠️ No API key found. Add GROQ_API_KEY to your .env or Streamlit secrets."
    prompt = f"{SYS_PROMPT}\n\nContext:\n{row_context(row)}"
    resp = completion(
        model=DEFAULT_MODEL,
        api_key=GROQ_API_KEY,
        messages=[{"role":"system","content":SYS_PROMPT},
                  {"role":"user","content":prompt}],
        temperature=0.2, max_tokens=220
    )
    return resp["choices"][0]["message"]["content"].strip()


st.title("LLM-Powered Investment Insight Generator")

col1, col2 = st.columns([2,1])
with col1:
    st.pyplot(plot_price_signals_sentiment(view), clear_figure=True)
with col2:
    if len(view):
        latest = view.iloc[-1]
        st.subheader("Daily Market Note")
        if st.button("Generate insight for latest date"):
            st.write(llm_insight_from_row(latest))

st.divider()
st.pyplot(plot_equity_curves(view), clear_figure=True)


num_cols = [c for c in ["ret_1d","avg_sentiment","rsi_14","macd","sma_cross"] if c in view.columns]
if len(num_cols) >= 2:
    corr = view[num_cols].corr()
    st.subheader("Correlation among key features")
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)));   ax.set_yticklabels(corr.index)
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
