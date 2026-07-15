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


@st.cache_data(ttl=60*60*6)
def get_risk_free_rate():
    import yfinance as yf
    data = yf.download("^TNX", period="5d", progress=False)
    rf = data["Close"].dropna().iloc[-1]

    if hasattr(rf, "iloc"):
        rf = rf.iloc[0]

    return float(rf) / 100

@st.cache_data(show_spinner=False)
def compute_portfolio_metrics(tickers, prices, weights, interval, tail):
    portfolio = Portfolio(tickers=tickers, df=prices, weights=weights, interval=interval, rf=get_risk_free_rate())
    summary = portfolio.uni_summary(tail=tail)
    returns = portfolio.returns().dropna()
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




def main():
    st.title("Portfolio Dashboard")
    st.caption("Analyze portfolio performance and risk through auto-calibrated simulations, stress testing, visualizations, and empirical statistics.")

    with st.sidebar:
        st.header("Portfolio inputs")
        tickers_input = st.text_input(
            "Tickers",
            value="GLD, XLE, VEA, META, JPM, SPY",
            help="Enter a comma-separated list of tickers, such as AAPL, MSFT, GOOGL.",
        )
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

        if not tickers:
            st.warning("Please enter at least one ticker.")
            st.stop()

        start_date = st.date_input("Start date", value=date.today() - timedelta(days=365 * 10))
        end_date = st.date_input("End date", value=date.today())

        interval = st.selectbox(
            "Time interval",
            options=["1d", "1wk", "1mo"],
            index=2,
            help="Choose the data frequency used for the analysis.",
        )

        # Default weights as equal-weighted
        num_tickers = len(tickers)
        default_weights = ", ".join([f"{1/num_tickers:.2f}" for _ in range(num_tickers)])
        
        weights_input = st.text_input(
            "Portfolio weights",
            value=default_weights,
            help="Optional. Enter comma-separated weights matching the ticker list, for example 0.5,0.3,0.2.",
        )
        st.caption("💡 Default: equal weights across all tickers")
        tail = st.slider("Tail risk quantile", min_value=0.005, max_value=0.2, value=0.01, step=0.005)

    try:
        weights = parse_weights(weights_input, tickers)
        prices = load_price_data(tickers, start_date, end_date, interval)
        portfolio, summary, returns = compute_portfolio_metrics(
            tickers=tickers,
            prices=prices,
            weights=weights,
            interval=interval,
            tail=tail,
        )
    except Exception as exc:
        st.error(f"Unable to build the dashboard: {exc}")
        st.stop()

    corr = portfolio.dependence(type="corr", tail=tail)

    st.markdown("**Portfolio Data:** \t\t *(choose inpnuts in sidebar)*")

    col1, col2, col3 = st.columns(3)
    col1.text("Securities: " + ", ".join(tickers))
    col2.text("Time Range: " + f"{start_date} to {end_date}")
    col3.text("Time Interval: " + interval)

    weights_array = np.array([weights.get(ticker, 0.0) for ticker in returns.columns])

    st.subheader("Tail Risk Simulation and Tail Dependence Stress Testing")
    st.markdown("**Methodology:**") 
    st.write(
        "Returns are assumed to be stationary, and are modeled through static probability distributions. " \
        "Marginal distributions are modeled by one of: Student-t, Laplace, GEV, or Normal families. " \
        "Dependence is modeled by a Student-t or Gaussian copula. The 'best' models are chosen based on lowest AIC. " \
        "Individual security returns are jointly simulated using the resulting random vector model, after which the implied portfolio returns are computed using the weights. " \
        "Log returns are used throughout the modeling and then converted back to percentage scale for the ouput."
    )
    col1, col2 = st.columns([0.55, 0.45])
    if len(tickers) > 1:
        with col2:
            st.markdown("**Apply Stress to Copula Correlations**")

            stress = st.slider(
                "Increase the magnitude of copula correlations by:",
                min_value=0,
                max_value=500,
                value=0,
                step=1,
                format="%d%%",
            )

            stressed_copula = portfolio.make_stressed_copula(
                stress_pct=stress,
                criterion="aic",
            )
            portfolio.set_copula(stressed_copula)

            assets = stressed_copula["assets"]
            stressed_corr = pd.DataFrame(
                stressed_copula["best_model"].sigma,
                index=assets,
                columns=assets,
            )

            st.caption(
                "Positive correlations become more positive and negative "
                "correlations become more negative."
            )

            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(6.5, 5))
            sns.heatmap(
                stressed_corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                cbar_kws={"label": "Copula correlation"},
                ax=ax,
            )
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col1:
            st.markdown("**Simulation Results**")

            try:
                import contextlib
                import io
                import matplotlib.pyplot as plt

                output = io.StringIO()

                with contextlib.redirect_stdout(output):
                    fig, mc_results = portfolio.monte_carlo_ES(
                        n_samples=int(1e5),
                        alpha=tail,
                    )

                if output.getvalue():
                    st.text(output.getvalue())

                st.pyplot(fig)
                plt.close(fig)

            except Exception as exc:
                st.error(f"Monte Carlo analysis failed: {exc}")

    else:
        st.info(
            "Monte Carlo copula analysis requires at least two tickers."
        )
    st.markdown(
"""
*Technical Notes*
- This will stress the correlation matrix *parameter* of the *copula* model, not the sample correlations. 
    - Generally, the copula allows for a much more flexible dependence model than the pearson correlation matrix of the sample, which only measures linear association.
- Correlations can only be tweaked such that the matrix remains a valid correlation matrix (i.e. positive semi-definite).
    - The stressed matrix is auto-adjusted so that it is valid.
"""
            )


    st.subheader("Portfolio overview")
    st.write(
        "The following tabs show various descriptive statistics using the empirical portfolio data."
    )

    st.subheader("Risk and return summary")
    summary_display = summary.copy()
    summary_display.index = [str(idx) for idx in summary_display.index]
    st.dataframe(summary_display, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cumulative Returns")
        try:
            import matplotlib.pyplot as plt
            
            # Calculate cumulative returns
            cum_rets = prices.apply(lambda x: 100 * (x / x.iloc[0] - 1))
            portfolio_cumulative = cum_rets.dot(weights_array) # prev: portfolio_returns = returns.dot(weights_array)
            #portfolio_cumulative = ((1 + portfolio_returns).cumprod() - 1) * 100
            
            fig, ax = plt.subplots(figsize=(10, 5))
            cum_rets.plot(
                ax=ax,
                title="Cumulative Returns: Individual Holdings vs Portfolio",
                xlabel="Date",
                ylabel="Cumulative Return (%)",
                alpha=0.5,
                legend=True
            )
            
            portfolio_cumulative.plot(
                ax=ax,
                linewidth=3,
                color="black",
                label="Portfolio"
            )
            
            ax.legend(loc='best')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"Cumulative returns plot unavailable: {exc}")
    with col2:
        st.subheader("Returns of Individual Holdings")
        try:
            import matplotlib.pyplot as plt

            individual_returns = prices.pct_change().dropna() * 100

            fig, ax = plt.subplots(figsize=(12, 6))

            individual_returns.plot(ax=ax)

            ax.set_title("Returns")
            ax.set_xlabel("Date")
            ax.set_ylabel("Return (%)")
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
            ax.legend(title="Security", loc="best")
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"Individual returns plot unavailable: {exc}")

    col1, col2 = st.columns(2)
    with col1:
        if len(tickers) > 1:
            st.subheader("Risk-Return Tradeoff")
            try:
                import matplotlib.pyplot as plt
                from matplotlib.ticker import MultipleLocator
                
                # Find the ES row dynamically based on the tail parameter
                es_label = f'{tail*100:.1f}% ES ({portfolio.interval})'
                
                fig, ax = plt.subplots(figsize=(6.5, 5))
                x = -summary.loc[es_label] 
                y = summary.loc['Mean (Annual)']
                ax.scatter(x=x, y=y, s=100, alpha=0.6)
                for col in summary.columns:
                    ax.text(x[col], y[col], col, fontsize=9, ha='center', va='bottom')
                ax.set_xlabel(f'Tail Risk (ES)')
                ax.set_ylabel('Mean Return (Annual)')
                ax.grid(True, alpha=0.3)
                rf = portfolio.rf
                ax.axhline(y=rf, color='blue', linestyle='--', linewidth=1, alpha=0.7)
                ax.annotate(
                    "10yr US Treasury Yield",
                    xy=(0.98, rf),
                    xycoords=("axes fraction", "data"),
                    ha="right",
                    va="bottom",
                    color="blue",
                    fontsize=9,
                )
                ax.yaxis.set_major_locator(MultipleLocator(0.05))
                fig.tight_layout()
                st.pyplot(fig)
                st.caption(
                    f"Expected Shortfall (ES) is reported on the same time scale selected in the sidebar"
                )
                plt.close(fig)
            except Exception as exc:
                st.warning(f"Risk-Return plot unavailable: {exc}")
    
    with col2:
        st.subheader("Correlation Matrix")
        if len(tickers) > 1:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                fig, ax = plt.subplots(figsize=(6.5, 5))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, cbar_kws={'label': 'Correlation'}, ax=ax)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as exc:
                st.warning(f"Correlation matrix unavailable: {exc}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("More Details on Methodology")
        st.markdown(
            r"""
    Let $R=(R_1,\ldots,R_d)^\top \in \mathbb{R}^d$ denote the vector
    of portfolio asset returns. Each margin is modeled using a parametric
    family, i.e.
    $$
    R_i \sim F_{\theta_i}, \quad i=1,\ldots,d,
    $$
    where $F_{\theta_i}$ is a CDF parameterized by $\theta_i$.
    The dependence structure is modeled through a copula $C:[0,1]^d\rightarrow[0,1]$, i.e. 
    we use the classical decompositions from Sklar's theorem, 
    $$
    F_R(r_1,\dots,r_d) = C\Big(F_{\theta_1}(r_1), \dots, F_{\theta_d}(r_d)\Big)
    $$
    $$
    R \overset{d}{=} \Big( F_{\theta_1}^{-1}(U_1), \dots, F_{\theta_d}^{-1}(U_d) \Big) \quad\text{for}\quad U=(U_1,\dots,U_d)\sim C
    $$
    where $F_R:\mathbb{R}^d\rightarrow[0,1]$ is the joint CDF of the random vector $R$. 
    The copula is estimated separately from the fitted margins 
    using rank-based pseudo-observations. Hence, this estimation framework is **semiparametric**. 
    In other words, just for the copula-fitting stage, we use the empirical CDF 
    $\widehat{F_i^\mathrm{emp}}$ to model each margin $i$. The fitted parametric margins 
    $\widehat{F_{\theta_i}}$ are used later, in the process of Monte Carlo, to transform 
    simulated copula uniforms back into individual security returns, nd finally 
    those resulting security returns and transformed into portfolio returns.
    """
        )

    with col2:
        st.subheader("Fitted Model Details")

        copula_result = portfolio.get_copula()

        st.write(
            f"Selected copula: *{copula_result['best_family']}* "
            f"using {copula_result['criterion'].upper()}."
        )

        st.dataframe(
            copula_result["summary"],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("*Fitted margins*")
        st.dataframe(
            portfolio.get_margins(),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
