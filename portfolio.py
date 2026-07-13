##########################################################################################
# portfolio.py
# Tools for portfolio analysis and risk management.
##########################################################################################

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import contextlib
import io
from copulae import GaussianCopula, StudentCopula
from scipy.stats import t as student_t, norm, laplace, genextreme


#------ Helper functions -----------------------------------------------------
def max_drawdown(prices):
    roll_max = prices.cummax()
    drawdown = prices / roll_max - 1
    return drawdown.min()

def num_params(cop, family):
    family = family.lower()
    d = cop.dim

    n_correlations = d * (d - 1) // 2

    if family == "gaussian":
        return n_correlations

    elif family == "student_t":
        return n_correlations + 1  # correlations + degrees of freedom

    else:
        raise ValueError(f"Unsupported copula family: {family}")
    
def extract_params(cop, family):
    family = family.lower()

    if family == "gaussian":
        rho_vec = np.asarray(cop.params)

        return {
            "rho": rho_vec.tolist(),
            "sigma": cop.sigma.tolist()
        }

    elif family == "student_t":
        rho_vec = np.asarray(cop.params.rho)

        return {
            "df": float(cop.params.df),
            "rho": rho_vec.tolist(),
            "sigma": cop.sigma.tolist()
        }

    else:
        raise ValueError(f"Unsupported copula family: {family}")


#------ Main class -----------------------------------------------------

class Portfolio:
    '''
    Fields:
        self.tickers is a list of security tickers in the portfolio
        self.weights (optional) is a dictionary with ticker keys and portfolio weight values
        self.data is a DataFrame of price data for the securities in the portfolio
        self.interval is the data interval (e.g., "1d", "1mo") used for the price data

    '''

    def __init__(self, tickers, df, weights={},interval="1d"):
        self.tickers = tickers
        self.weights = weights
        self.data = df
        self.interval = interval

    def holding_data(self, holding):
        df = self.data
        if holding != 'all':
            df = df[holding]
        return df


    def returns(self, holding='all', log=True):
        '''
        The log returns L[t] = log(P[t]/P[t-1]) = log(P[t]) - log(P[t-1])
        or the ordinary returns P[t]/P[t-1] - 1 if log=False.
        '''
        df = self.holding_data(holding)
        if log:
            return np.log(df).diff().dropna()
        else:
            return df.pct_change().dropna()
        

    def uni_summary(self, holding='all', log=True, tail=0.05):
        '''
        Returns a univariate summary of the returns of each security (individually), including:
            - the annualized mean and volatility, 
            - the skew and excess kurtosis, and 
            - some tail risk metrics (VaR, ES/CVaR, MaxDD).
        '''
        df = self.returns(holding=holding, log=log)
        means = df.mean()

        scale = 252 if self.interval == "1d" else 12 if self.interval == "1mo" else 1

        d = {}
        if log:
            d['Mean (Ann.)'] = means * scale
        else: 
            d['Mean (Ann.)'] = (1 + means)**scale - 1 
            # Note: this is equivalent to the expression in the log=True case
            #       although we break up the cases to take advantage of the
            #       simplified formula for log=True.
        d['Volatility (Ann.)'] = df.std() * np.sqrt(scale)
        d['Skew'] = df.skew()
        d['Kurtosis (Excess)'] = df.kurtosis()
        d[f'VaR({tail*100:.1f}%)'] = df.quantile(tail) 
        d[f'ES({tail*100:.1f}%)'] = df.apply(lambda col: col[col <= col.quantile(tail)].mean())
        d[f'MaxDD'] = max_drawdown( (1+df).cumprod() )

        return pd.DataFrame(d).T
    

    def dependence(self, type="corr", holding="all", log=True, tail=0.05):
        """
        Dependence measures between return series.

        type options:
            - "corr": Pearson correlation
            - "cov": covariance
            - "spearman": rank correlation
            - "kendall": Kendall's tau
            - "lower_tail": empirical lower tail dependence
            - "upper_tail": empirical upper tail dependence
            - "all": dictionary of several dependence measures
        """

        df = self.returns(holding=holding, log=log).dropna()

        if type == "corr":
            return df.corr(method="pearson")
        elif type == "cov":
            return df.cov()
        elif type == "spearman":
            return df.corr(method="spearman")
        elif type == "kendall":
            return df.corr(method="kendall")
        elif type in ["lower_tail", "upper_tail"]:
            ranks = df.rank(method="average") / (len(df) + 1)
            out = pd.DataFrame(
                np.eye(len(df.columns)),
                index=df.columns,
                columns=df.columns
            )
            for i in df.columns:
                for j in df.columns:
                    if i == j:
                        continue
                    if type == "lower_tail":
                        # P(U_i <= q | U_j <= q)
                        numerator = ((ranks[i] <= tail) & (ranks[j] <= tail)).mean()
                        denominator = (ranks[j] <= tail).mean()
                    else:
                        # P(U_i >= 1-q | U_j >= 1-q)
                        numerator = ((ranks[i] >= 1 - tail) & (ranks[j] >= 1 - tail)).mean()
                        denominator = (ranks[j] >= 1 - tail).mean()
                    out.loc[i, j] = numerator / denominator if denominator > 0 else np.nan
            return out
        elif type == "all":
            return {
                "pearson_corr": df.corr(method="pearson"),
                "spearman_corr": df.corr(method="spearman"),
                "kendall_tau": df.corr(method="kendall"),
                "covariance": df.cov(),
                f"lower_tail_{tail}": self.dependence(
                    type="lower_tail", holding=holding, log=log, tail=tail
                ),
                f"upper_tail_{tail}": self.dependence(
                    type="upper_tail", holding=holding, log=log, tail=tail
                )
            }
        else:
            raise ValueError(
                "type argument must be one of: 'corr', 'cov', 'spearman', 'kendall', "
                "'lower_tail', 'upper_tail', or 'all'"
            )
        
    def margin_fit(self, holding="all", log=True): 
        """
        Fit marginal distributions to individual security returns. 

        Families:
            - Normal
            - Student-t
            - Laplace (Double Exponential)
            - GEV (Generalized Extreme Value)
        """
        
        df = self.returns(holding=holding, log=log).dropna()

        if df.shape[1] < 1:
            raise ValueError("Need at least one asset to fit marginals.")

        results = []

        for col in df.columns:
            series = df[col].dropna()
            n = len(series)

            # Fit Normal distribution
            mu, sigma = series.mean(), series.std()
            ll_normal = np.sum(-0.5 * np.log(2 * np.pi * sigma**2) - ((series - mu)**2) / (2 * sigma**2))
            aic_normal = -2 * ll_normal + 2 * 2  # 2 parameters: mu and sigma
            bic_normal = -2 * ll_normal + np.log(n) * 2

            # Fit Student-t distribution
            params_t = student_t.fit(series)
            ll_t = np.sum(student_t.logpdf(series, *params_t))
            aic_t = -2 * ll_t + 2 * len(params_t)
            bic_t = -2 * ll_t + np.log(n) * len(params_t)

            # Fit Laplace distribution
            params_laplace = laplace.fit(series)
            ll_laplace = np.sum(laplace.logpdf(series, *params_laplace))
            aic_laplace = -2 * ll_laplace + 2 *len(params_laplace)
            bic_laplace = -2 * ll_laplace + np.log(n) * len(params_laplace)

            # Fit GEV distribution
            params_gev = genextreme.fit(series)
            ll_gev = np.sum(genextreme.logpdf(series, *params_gev))
            aic_gev = -2 * ll_gev + 2 * len(params_gev)
            bic_gev = -2 * ll_gev + np.log(n) * len(params_gev) 

            results.append({
                "Asset": col,
                "Normal": {"mu": mu, "sigma": sigma, "loglik": ll_normal, "aic": aic_normal, "bic": bic_normal},
                "Student-t": {"params": params_t, "loglik": ll_t, "aic": aic_t, "bic": bic_t},
                "Laplace": {"params": params_laplace, "loglik": ll_laplace, "aic": aic_laplace, "bic": bic_laplace}, 
                "GEV": {"params": params_gev, "loglik": ll_gev, "aic": aic_gev, "bic": bic_gev},
                "Best-Fit-AIC": None,
                "Best-Fit-BIC": None
            })

            results[-1]["Best-Fit-AIC"] = min(["Normal", "Student-t", "Laplace", "GEV"], key=lambda x: results[-1][x]["aic"])
            results[-1]["Best-Fit-BIC"] = min(["Normal", "Student-t", "Laplace", "GEV"], key=lambda x: results[-1][x]["bic"])

        self._margins = pd.DataFrame(results)
        return self._margins
    
    def copula_fit(self, holding="all", log=True, criterion="bic"):
        """
        Fit a copula to portfolio returns.

        Families:
            - Gaussian
            - Student-t
        """

        df = self.returns(holding=holding, log=log).dropna()

        if df.shape[1] < 2:
            raise ValueError("Need at least two assets to fit a copula.")

        criterion = criterion.lower()
        if criterion not in ["aic", "bic"]:
            raise ValueError("criterion must be 'aic' or 'bic'.")

        d = df.shape[1]

        models = {
            "gaussian": GaussianCopula(dim=d),
            "student_t": StudentCopula(dim=d)
        }

        results = []

        for name, cop in models.items():
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    cop.fit(df, to_pobs=True)

                u = cop.pobs(df)
                ll = cop.log_lik(u, to_pobs=False)
                k = num_params(cop, name)
                n = len(df)

                aic = -2 * ll + 2 * k
                bic = -2 * ll + np.log(n) * k

                params = extract_params(cop, name)

                results.append({
                    "family": name,
                    "loglik": ll,
                    "n_params": k,
                    "aic": aic,
                    "bic": bic,
                    "params": params,
                    "model": cop,
                    "error": None
                })

            except Exception as e:
                results.append({
                    "family": name,
                    "loglik": np.nan,
                    "n_params": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                    "params": None,
                    "model": None,
                    "error": repr(e)
                })

        summary = (
            pd.DataFrame(results)
            .drop(columns=["model"], errors="ignore")
            .sort_values(criterion, na_position="last")
            .reset_index(drop=True)
        )

        valid = [r for r in results if r["model"] is not None]

        if not valid:
            raise RuntimeError("No copula models successfully fit.")

        best = min(valid, key=lambda r: r[criterion])

        self._copula = {
            "summary": summary,
            "best_family": best["family"],
            "best_params": best["params"],
            "best_model": best["model"],
            "all_results": results,
            "assets": list(df.columns),
            "n_obs": len(df),
            "dimension": d,
            "criterion": criterion
        }
        return self._copula
    
    def joint_simulator(self, holding="all", criterion="aic", margins=None, copula=None):
        """
        Build and return a callable that simulates joint security returns.

        The returned function first draws dependent uniforms from the fitted
        copula and then transforms each column through the PPF of that asset's
        selected marginal distribution.

        Log returns are used.
        """
        criterion = criterion.lower()

        if margins is None:
            if not hasattr(self, "_margins"):
                margins = self.margin_fit(holding=holding, log=True)
            else:
                margins = self._margins

        if copula is None:
            if not hasattr(self, "_copula"):
                copula = self.copula_fit(
                    holding=holding,
                    log=True,
                    criterion=criterion
                )
            else:
                copula = self._copula

        if holding == "all":
            assets = list(copula.get("assets", self.data.columns))
        elif isinstance(holding, str):
            assets = [holding]
        else:
            assets = list(holding)

        copula_assets = list(copula.get("assets", assets))
        if assets != copula_assets:
            raise ValueError(
                "The requested holdings must exactly match the assets and "
                "column order used to fit the copula. Refit the copula for "
                "the requested holdings before constructing the simulator."
            )

        margin_rows = margins.set_index("Asset", drop=False)
        missing = [asset for asset in assets if asset not in margin_rows.index]
        if missing:
            raise ValueError(f"Missing fitted margins for assets: {missing}")

        ppf_map = {
            "Normal": norm.ppf,
            "Student-t": student_t.ppf,
            "Laplace": laplace.ppf,
            "GEV": genextreme.ppf
        }

        marginal_specs = []
        selection_column = f"Best-Fit-{criterion.upper()}"

        for asset in assets:
            row = margin_rows.loc[asset]
            family = row[selection_column]
            fit_info = row[family]
            
            if "params" in fit_info:
                params = tuple(fit_info["params"])
            elif family == "Normal":
                params = (float(fit_info["mu"]), float(fit_info["sigma"]))
            else:
                raise ValueError(
                    f"No fitted parameter tuple found for {asset} ({family})."
                )

            marginal_specs.append((asset, family, ppf_map[family], params))

        copula_model = copula.get("best_model")
        if copula_model is None:
            raise ValueError(
                "The copula result does not contain 'best_model'. Keep the "
                "fitted model returned by copula_fit, or reconstruct it before "
                "calling joint_simulator."
            )

        dimension = len(assets)
        fitted_dimension = int(copula.get("dimension", dimension))
        if dimension != fitted_dimension:
            raise ValueError(
                f"Copula dimension is {fitted_dimension}, but {dimension} "
                "margins were selected."
            )

        def simulate(n_samples=1, random_state=None, u=None):
            """Simulate joint return vectors from the fitted model."""
            if u is None:                   
                if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
                    raise ValueError("n_samples must be a positive integer.")

                # copulae uses NumPy's global RNG internally in some versions.
                # Preserve global state when a reproducible local seed is given.
                if random_state is None:
                    uniforms = copula_model.random(int(n_samples))
                else:
                    state = np.random.get_state()
                    try:
                        np.random.seed(random_state)
                        uniforms = copula_model.random(int(n_samples))
                    finally:
                        np.random.set_state(state)
            else:
                uniforms = np.asarray(u, dtype=float)
                if uniforms.ndim == 1:
                    uniforms = uniforms.reshape(1, -1)
                if uniforms.ndim != 2:
                    raise ValueError("u must be a one- or two-dimensional array.")
                n_samples = uniforms.shape[0]

            uniforms = np.asarray(uniforms, dtype=float)
            if uniforms.ndim == 1:
                uniforms = uniforms.reshape(1, -1)

            simulated = np.empty_like(uniforms, dtype=float)

            for j, (_, _, ppf, params) in enumerate(marginal_specs):
                simulated[:, j] = ppf(uniforms[:, j], *params)

            return pd.DataFrame(simulated, columns=assets)

        # Helpful metadata without changing the callable interface.
        simulate.assets = tuple(assets)
        simulate.margin_criterion = criterion
        simulate.margin_families = {
            asset: family for asset, family, _, _ in marginal_specs
        }
        simulate.copula_family = copula.get("best_family")

        self._simulator = simulate
        return simulate
    
    def monte_carlo_ES(self, n_samples=int(1e4), alpha=0.05, random_state=None):
        """
        Estimate the portfolio Expected Shortfall (ES) at a given confidence level
        using Monte Carlo simulation based on the fitted copula and marginal distributions.

        Parameters:
            n_samples: Number of Monte Carlo samples to generate.
            alpha: Significance level for ES (e.g., 0.05 for 5% ES).
            random_state: Seed for reproducibility.
        """
        if not hasattr(self, "_simulator"):
            simulate = self.joint_simulator()

        simulate = self._simulator
        simulated_returns = simulate(n_samples=n_samples, random_state=random_state)

        if not self.weights:
            raise ValueError("Portfolio weights are not defined.")

        weights_array = np.array([self.weights.get(asset, 0) for asset in simulated_returns.columns])
        portfolio_returns = simulated_returns.dot(weights_array) 
        portfolio_returns = np.exp(portfolio_returns) - 1 # convert to % scale from log
        var_threshold = np.quantile(portfolio_returns, alpha) 
        es_array = portfolio_returns[portfolio_returns <= var_threshold]

        es_estimate = es_array.mean()
        es_se = es_array.std() / np.sqrt(n_samples)

        print(f"Monte Carlo estimate of {100*alpha:.2f}% expected shortfall (ES) is ${100*es_estimate:.2f}%$")
        print(f"Standard Error of ES estimate: ${100*es_se:.6f}%$")

        fig = plt.figure(figsize=(12, 8))
        portfolio_returns.hist(bins=50, density=True, alpha=0.5, color='grey')
        plt.axvline(es_estimate, color='red', linestyle='dashed', linewidth=2, label=f'ES({100*alpha:.2f}%)=${100*es_estimate:.2f}%$')
        plt.axvline(var_threshold, color='orange', linestyle='dashed', linewidth=2, label=f'VaR({100*alpha:.2f}%)=${100*var_threshold:.2f}%$')
        plt.axvline(portfolio_returns.mean(), color='green', linestyle='dashed', linewidth=2, label=f'Mean=${100*portfolio_returns.mean():.2f}%$')
        skew = portfolio_returns.skew()
        kurtosis = portfolio_returns.kurtosis()
        plt.text(
            0.98, 0.95,
            f"Skew: {skew:.3f}",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            color="purple"
        )
        plt.text(
            0.98, 0.90,
            f"Excess Kurtosis: {kurtosis:.2f}",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            color="brown"
        )
        plt.legend()
        plt.title(f"Monte Carlo Simulation of Portfolio Returns (n={n_samples})")
        plt.xlabel("Portfolio Returns")
        plt.ylabel("Frequency")

        plt.legend()
        plt.title(f"Monte Carlo Simulation of Portfolio Returns (n={n_samples})")
        plt.xlabel("Portfolio Returns")
        plt.ylabel("Density")

        return fig, pd.DataFrame({
            "Est": [es_estimate],
            "SE": [es_se]
            })



