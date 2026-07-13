from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from portfolio import Portfolio


st.set_page_config(page_title="Portfolio Metrics Dashboard", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load_price_data(tickers, start_date, end_date, interval):
    import yfinance as yf
    
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date + timedelta(days=1),
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError("No price data returned for the selected tickers and date range.")

    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data["Close"]
        except KeyError:
            data = data.xs("Close", axis=1, level=0)

    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)

    data.columns = [str(col) for col in data.columns]
    return data.sort_index()


@st.cache_data(show_spinner=False)
def compute_portfolio_metrics(tickers, prices, weights, interval, log_returns, tail):
    portfolio = Portfolio(tickers=tickers, df=prices, weights=weights, interval=interval)
    summary = portfolio.uni_summary(log=log_returns, tail=tail)
    returns = portfolio.returns(log=log_returns).dropna()
    return portfolio, summary, returns


def parse_weights(raw_weights, tickers):
    if raw_weights is None or str(raw_weights).strip() == "":
        return {ticker: 1 / len(tickers) for ticker in tickers}

    values = [value.strip() for value in raw_weights.split(",") if value.strip()]
    if len(values) != len(tickers):
        raise ValueError("The number of weights must match the number of tickers.")

    weights = {}
    for ticker, value in zip(tickers, values):
        weights[ticker] = float(value)
    return weights


def normalize_prices(prices):
    return prices / prices.iloc[0]


def main():
    st.title("Portfolio Metrics Dashboard")
    st.caption("Analyze portfolio risk, return behavior, and dependence with the project’s existing analytics code.")

    with st.sidebar:
        st.header("Portfolio inputs")
        tickers_input = st.text_input(
            "Tickers",
            value="AAPL, MSFT, GOOGL",
            help="Enter a comma-separated list of tickers, such as AAPL, MSFT, GOOGL.",
        )
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

        if not tickers:
            st.warning("Please enter at least one ticker.")
            st.stop()

        start_date = st.date_input("Start date", value=date.today() - timedelta(days=365 * 3))
        end_date = st.date_input("End date", value=date.today())

        interval = st.selectbox(
            "Time interval",
            options=["1d", "1wk", "1mo"],
            index=0,
            help="Choose the data frequency used for the analysis.",
        )

        weights_input = st.text_input(
            "Portfolio weights",
            value="",
            help="Optional. Enter comma-separated weights matching the ticker list, for example 0.5,0.3,0.2.",
        )
        log_returns = st.checkbox("Use log returns", value=True)
        tail = st.slider("Tail risk quantile", min_value=0.01, max_value=0.2, value=0.05, step=0.01)

    try:
        weights = parse_weights(weights_input, tickers)
        prices = load_price_data(tickers, start_date, end_date, interval)
        portfolio, summary, returns = compute_portfolio_metrics(
            tickers=tickers,
            prices=prices,
            weights=weights,
            interval=interval,
            log_returns=log_returns,
            tail=tail,
        )
    except Exception as exc:
        st.error(f"Unable to build the dashboard: {exc}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Tickers", ", ".join(tickers))
    col2.metric("Date range", f"{start_date} → {end_date}")
    col3.metric("Interval", interval)

    st.subheader("Monte Carlo tail-risk preview")
    if len(tickers) > 1:
        try:
            import matplotlib.pyplot as plt
            simulator = portfolio.joint_simulator(holding="all", criterion="bic")
            mc_returns = simulator(n_samples=2000, random_state=42)
            weights_array = np.array([weights.get(ticker, 0.0) for ticker in returns.columns])
            mc_portfolio = mc_returns.dot(weights_array)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(mc_portfolio, bins=50, density=True, alpha=0.6, color="steelblue", edgecolor="black")
            ax.axvline(np.quantile(mc_portfolio, 0.05), color="red", linestyle="--", linewidth=2, label="5% VaR")
            ax.axvline(mc_portfolio.mean(), color="green", linestyle="--", linewidth=2, label="Mean")
            ax.set_title("Simulated portfolio return distribution", fontsize=14, fontweight="bold")
            ax.set_xlabel("Portfolio return", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.legend(loc="upper left", fontsize=11)
            fig.tight_layout()
            st.pyplot(fig)
        except Exception as exc:
            st.info(f"Monte Carlo preview is unavailable for this selection: {exc}")
    else:
        st.info("Monte Carlo tail-risk preview requires at least two tickers.")

    st.subheader("Portfolio overview")
    st.write(
        "This screen uses the existing portfolio analysis module to summarize returns, volatility, tail risk, and portfolio-level behavior."
    )

    st.subheader("Price evolution")
    normalized = normalize_prices(prices)
    st.line_chart(normalized)

    st.subheader("Risk and return summary")
    summary_display = summary.copy()
    summary_display.index = [str(idx) for idx in summary_display.index]
    st.dataframe(summary_display, use_container_width=True)

    if len(tickers) > 1:
        st.subheader("Correlation matrix")
        corr = portfolio.dependence(type="corr", log=log_returns, tail=tail)
        st.dataframe(corr, use_container_width=True)

    st.subheader("Weighted portfolio returns")
    weights_array = np.array([weights.get(ticker, 0.0) for ticker in returns.columns])
    portfolio_returns = returns.dot(weights_array)
    st.line_chart(pd.DataFrame({"Portfolio Returns": (1 + portfolio_returns).cumprod()}))

    st.subheader("Portfolio weights")
    weights_df = pd.DataFrame({"Ticker": tickers, "Weight": [weights.get(ticker, 0.0) for ticker in tickers]})
    st.dataframe(weights_df, use_container_width=True)


if __name__ == "__main__":
    main()
