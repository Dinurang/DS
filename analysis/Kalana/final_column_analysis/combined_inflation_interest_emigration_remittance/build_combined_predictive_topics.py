from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent
INPUT_CSV = REPO_ROOT / "Dataset_Management" / "SriLanka_Migration_final.csv"
CHART_DIR = OUT_DIR / "charts"

RANDOM_STATE = 42


def load_annual_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Invalid date values found in source CSV.")

    cols = [
        "slbfe_total_annual",
        "remittances_annual_usd_mn",
        "inflation_rate_annual",
        "central_bank_interest_rate_annual",
    ]
    annual = (
        df.assign(year=df["date"].dt.year)
        .groupby("year", as_index=False)[cols]
        .mean()
        .sort_values("year")
        .reset_index(drop=True)
    )
    return annual


def make_eda_plots(annual: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(annual["year"], annual["inflation_rate_annual"], marker="o", linewidth=2.0, color="#C0392B")
    axes[0].set_title("Annual Inflation Rate (1994-2025)")
    axes[0].set_ylabel("Inflation rate (%)")

    axes[1].plot(
        annual["year"],
        annual["central_bank_interest_rate_annual"],
        marker="o",
        linewidth=2.0,
        color="#1F618D",
    )
    axes[1].set_title("Annual Central Bank Interest Rate (1994-2025)")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Interest rate (%)")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "3_1_eda_trends.png", dpi=180)
    plt.close(fig)

    corr = annual[
        [
            "slbfe_total_annual",
            "remittances_annual_usd_mn",
            "inflation_rate_annual",
            "central_bank_interest_rate_annual",
        ]
    ].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0.0, ax=ax)
    ax.set_title("Correlation Heatmap (Annual Aggregation)")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "3_1_eda_correlation_heatmap.png", dpi=180)
    plt.close(fig)



def evaluate_models(annual: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str]]:
    X = annual[["slbfe_total_annual", "remittances_annual_usd_mn"]].to_numpy()
    targets = {
        "inflation_rate_annual": annual["inflation_rate_annual"].to_numpy(),
        "central_bank_interest_rate_annual": annual["central_bank_interest_rate_annual"].to_numpy(),
    }
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    loo = LeaveOneOut()
    rows: list[dict[str, object]] = []
    best_predictions: dict[str, np.ndarray] = {}
    best_model_name: dict[str, str] = {}

    for target_name, y in targets.items():
        best_rmse = None
        for model_name, model in models.items():
            y_pred = cross_val_predict(model, X, y, cv=loo)
            mae = mean_absolute_error(y, y_pred)
            rmse = mean_squared_error(y, y_pred) ** 0.5
            r2 = r2_score(y, y_pred)
            rows.append(
                {
                    "target": target_name,
                    "model": model_name,
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "r2": float(r2),
                    "n_samples": int(len(y)),
                    "validation": "Leave-One-Out CV",
                }
            )
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_predictions[target_name] = y_pred
                best_model_name[target_name] = model_name

    metrics = pd.DataFrame(rows)
    return metrics, best_predictions, best_model_name



def make_ml_plots(annual: pd.DataFrame, metrics: pd.DataFrame, best_predictions: dict[str, np.ndarray], best_model_name: dict[str, str]) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, target in zip(
        axes,
        ["inflation_rate_annual", "central_bank_interest_rate_annual"],
        strict=True,
    ):
        sub = metrics[metrics["target"] == target].copy()
        sub = sub.sort_values("rmse", ascending=True)
        sns.barplot(data=sub, x="model", y="rmse", hue="model", ax=ax, palette="viridis", legend=False)
        ax.set_title(f"RMSE by Model: {target}")
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(CHART_DIR / "3_3_2_evaluation_metrics_rmse.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, target in zip(
        axes,
        ["inflation_rate_annual", "central_bank_interest_rate_annual"],
        strict=True,
    ):
        y_true = annual[target].to_numpy()
        y_pred = best_predictions[target]
        ax.scatter(y_true, y_pred, alpha=0.8, color="#2E86C1")
        min_v = float(min(y_true.min(), y_pred.min()))
        max_v = float(max(y_true.max(), y_pred.max()))
        ax.plot([min_v, max_v], [min_v, max_v], color="#C0392B", linewidth=2)
        ax.set_title(f"Observed vs Predicted ({best_model_name[target]})")
        ax.set_xlabel("Observed")
        ax.set_ylabel("LOOCV Predicted")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "3_3_3_ml_results_observed_vs_predicted.png", dpi=180)
    plt.close(fig)



def run_statistical_inference(annual: pd.DataFrame) -> dict[str, object]:
    X = annual[["slbfe_total_annual", "remittances_annual_usd_mn"]]
    X_const = sm.add_constant(X)

    y_inf = annual["inflation_rate_annual"]
    model_inf = sm.OLS(y_inf, X_const).fit()

    y_rate = annual["central_bank_interest_rate_annual"]
    model_rate = sm.OLS(y_rate, X_const).fit()

    rho_inf, p_inf = spearmanr(annual["remittances_annual_usd_mn"], annual["inflation_rate_annual"])
    rho_rate, p_rate = spearmanr(annual["slbfe_total_annual"], annual["central_bank_interest_rate_annual"])

    out = {
        "n_years": int(len(annual)),
        "year_start": int(annual["year"].min()),
        "year_end": int(annual["year"].max()),
        "hypothesis_1_inflation_model": {
            "f_statistic": float(model_inf.fvalue),
            "f_pvalue": float(model_inf.f_pvalue),
            "r_squared": float(model_inf.rsquared),
            "adj_r_squared": float(model_inf.rsquared_adj),
            "coef_const": float(model_inf.params["const"]),
            "coef_slbfe_total_annual": float(model_inf.params["slbfe_total_annual"]),
            "coef_remittances_annual_usd_mn": float(model_inf.params["remittances_annual_usd_mn"]),
            "p_slbfe_total_annual": float(model_inf.pvalues["slbfe_total_annual"]),
            "p_remittances_annual_usd_mn": float(model_inf.pvalues["remittances_annual_usd_mn"]),
            "ci_slbfe_total_annual": [float(x) for x in model_inf.conf_int().loc["slbfe_total_annual"].to_list()],
            "ci_remittances_annual_usd_mn": [float(x) for x in model_inf.conf_int().loc["remittances_annual_usd_mn"].to_list()],
            "spearman_rho_remittance_inflation": float(rho_inf),
            "spearman_p_remittance_inflation": float(p_inf),
        },
        "hypothesis_2_interest_model": {
            "f_statistic": float(model_rate.fvalue),
            "f_pvalue": float(model_rate.f_pvalue),
            "r_squared": float(model_rate.rsquared),
            "adj_r_squared": float(model_rate.rsquared_adj),
            "coef_const": float(model_rate.params["const"]),
            "coef_slbfe_total_annual": float(model_rate.params["slbfe_total_annual"]),
            "coef_remittances_annual_usd_mn": float(model_rate.params["remittances_annual_usd_mn"]),
            "p_slbfe_total_annual": float(model_rate.pvalues["slbfe_total_annual"]),
            "p_remittances_annual_usd_mn": float(model_rate.pvalues["remittances_annual_usd_mn"]),
            "ci_slbfe_total_annual": [float(x) for x in model_rate.conf_int().loc["slbfe_total_annual"].to_list()],
            "ci_remittances_annual_usd_mn": [float(x) for x in model_rate.conf_int().loc["remittances_annual_usd_mn"].to_list()],
            "spearman_rho_emigration_interest": float(rho_rate),
            "spearman_p_emigration_interest": float(p_rate),
        },
    }
    return out



def make_inference_plot(annual: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.regplot(
        data=annual,
        x="remittances_annual_usd_mn",
        y="inflation_rate_annual",
        ci=95,
        scatter_kws={"alpha": 0.8, "s": 45},
        line_kws={"color": "#C0392B", "linewidth": 2},
        ax=axes[0],
    )
    axes[0].set_title("Remittances vs Inflation (Annual)")
    axes[0].set_xlabel("Remittances annual (USD mn)")
    axes[0].set_ylabel("Inflation rate annual (%)")

    sns.regplot(
        data=annual,
        x="slbfe_total_annual",
        y="central_bank_interest_rate_annual",
        ci=95,
        scatter_kws={"alpha": 0.8, "s": 45},
        line_kws={"color": "#1F618D", "linewidth": 2},
        ax=axes[1],
    )
    axes[1].set_title("Emigration vs Central Bank Interest Rate (Annual)")
    axes[1].set_xlabel("SLBFE total annual emigration")
    axes[1].set_ylabel("Central bank interest rate annual (%)")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "3_3_4_statistical_inference_relationships.png", dpi=180)
    plt.close(fig)



def write_latex_block(inference: dict[str, object]) -> None:
    h1 = inference["hypothesis_1_inflation_model"]
    h2 = inference["hypothesis_2_interest_model"]
    n = inference["n_years"]
    y0 = inference["year_start"]
    y1 = inference["year_end"]
    h1_joint = "rejected" if float(h1["f_pvalue"]) < 0.05 else "not rejected"
    h2_joint = "rejected" if float(h2["f_pvalue"]) < 0.05 else "not rejected"
    h1_sig = "significant" if float(h1["f_pvalue"]) < 0.05 else "not significant"
    h2_sig = "significant" if float(h2["f_pvalue"]) < 0.05 else "not significant"

    latex = rf"""\indent \textbf{{Hypothesis 1: Joint Association of Emigration and Remittances with Inflation}}\\
We investigated whether annual inflation is jointly associated with emigration volume (\texttt{{slbfe\_total\_annual}}) and remittance inflows (\texttt{{remittances\_\allowbreak annual\_\allowbreak usd\_\allowbreak mn}}) over {y0}--{y1}. The inferential model was multiple OLS regression with annual observations ($n = {n}$):
\[
y_t = \beta_0 + \beta_1 x_{{1,t}} + \beta_2 x_{{2,t}} + \epsilon_t
\]
where $y_t = \texttt{{inflation\_rate\_annual}}_t$, $x_{{1,t}} = \texttt{{slbfe\_total\_annual}}_t$, and $x_{{2,t}} = \texttt{{remittances\_\allowbreak annual\_\allowbreak usd\_\allowbreak mn}}_t$.
Inference was conducted at $\alpha = 0.05$.
\begin{{itemize}}[noitemsep, topsep=0pt]
    \item \textbf{{Null Hypothesis}} ($H_0$): $\beta_1 = \beta_2 = 0$ (no joint linear association with inflation).
    \item \textbf{{Alternative Hypothesis}} ($H_1$): At least one slope is non-zero ($\beta_1 \neq 0$ or $\beta_2 \neq 0$).
\end{{itemize}}
\noindent \textbf{{Results:}} The overall model was {h1_sig}, $F = {h1['f_statistic']:.3f}$, $p = {h1['f_pvalue']:.4f}$, with $R^2 = {h1['r_squared']:.3f}$ (adjusted $R^2 = {h1['adj_r_squared']:.3f}$). Coefficient estimates were: $\hat\beta_1 = {h1['coef_slbfe_total_annual']:.8f}$ ($p = {h1['p_slbfe_total_annual']:.4f}$; 95\% CI [{h1['ci_slbfe_total_annual'][0]:.8f}, {h1['ci_slbfe_total_annual'][1]:.8f}]) and $\hat\beta_2 = {h1['coef_remittances_annual_usd_mn']:.6f}$ ($p = {h1['p_remittances_annual_usd_mn']:.4f}$; 95\% CI [{h1['ci_remittances_annual_usd_mn'][0]:.6f}, {h1['ci_remittances_annual_usd_mn'][1]:.6f}]). A non-parametric sensitivity check gave Spearman $\rho = {h1['spearman_rho_remittance_inflation']:.3f}$ ($p = {h1['spearman_p_remittance_inflation']:.4f}$) for remittances vs inflation.

\noindent \textbf{{Interpretation:}} At $\alpha = 0.05$, $H_0$ is {h1_joint} for the joint model. The pair of migration-finance predictors shows limited joint explanatory power for inflation at conventional significance, while remittances retain a statistically significant negative coefficient in this specification.
~\\

\indent \textbf{{Hypothesis 2: Joint Association of Emigration and Remittances with Central Bank Interest Rate}}\\
We investigated whether annual central bank interest rates are jointly associated with emigration volume and remittance inflows across {y0}--{y1}. The multiple OLS model was:
\[
y_t = \gamma_0 + \gamma_1 x_{{1,t}} + \gamma_2 x_{{2,t}} + u_t
\]
where $y_t = \texttt{{central\_bank\_interest\_rate\_annual}}_t$, $x_{{1,t}} = \texttt{{slbfe\_total\_annual}}_t$, and $x_{{2,t}} = \texttt{{remittances\_\allowbreak annual\_\allowbreak usd\_\allowbreak mn}}_t$.
Inference was conducted at $\alpha = 0.05$.
\begin{{itemize}}[noitemsep, topsep=0pt]
    \item \textbf{{Null Hypothesis}} ($H_0$): $\gamma_1 = \gamma_2 = 0$ (no joint linear association with interest rate).
    \item \textbf{{Alternative Hypothesis}} ($H_1$): At least one slope is non-zero ($\gamma_1 \neq 0$ or $\gamma_2 \neq 0$).
\end{{itemize}}
\noindent \textbf{{Results:}} The overall model was {h2_sig}, $F = {h2['f_statistic']:.3f}$, $p = {h2['f_pvalue']:.4f}$, with $R^2 = {h2['r_squared']:.3f}$ (adjusted $R^2 = {h2['adj_r_squared']:.3f}$). Coefficient estimates were: $\hat\gamma_1 = {h2['coef_slbfe_total_annual']:.8f}$ ($p = {h2['p_slbfe_total_annual']:.4f}$; 95\% CI [{h2['ci_slbfe_total_annual'][0]:.8f}, {h2['ci_slbfe_total_annual'][1]:.8f}]) and $\hat\gamma_2 = {h2['coef_remittances_annual_usd_mn']:.6f}$ ($p = {h2['p_remittances_annual_usd_mn']:.4f}$; 95\% CI [{h2['ci_remittances_annual_usd_mn'][0]:.6f}, {h2['ci_remittances_annual_usd_mn'][1]:.6f}]). A sensitivity check gave Spearman $\rho = {h2['spearman_rho_emigration_interest']:.3f}$ ($p = {h2['spearman_p_emigration_interest']:.4f}$) for emigration vs interest rate.

\noindent \textbf{{Interpretation:}} At $\alpha = 0.05$, $H_0$ is {h2_joint} for the joint model. Emigration and remittance levels together have statistically detectable explanatory value for annual central bank interest-rate variation, but causal interpretation is not warranted from this observational analysis alone.
~\\
"""
    (OUT_DIR / "statistical_inference_latex.tex").write_text(latex, encoding="utf-8")



def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    annual = load_annual_data()
    annual.to_csv(OUT_DIR / "annual_aggregated_1994_2025.csv", index=False)

    make_eda_plots(annual)
    metrics, best_predictions, best_model_name = evaluate_models(annual)
    metrics.to_csv(OUT_DIR / "ml_metrics_loocv.csv", index=False)
    make_ml_plots(annual, metrics, best_predictions, best_model_name)

    inference = run_statistical_inference(annual)
    (OUT_DIR / "inference_results.json").write_text(json.dumps(inference, indent=2), encoding="utf-8")
    make_inference_plot(annual)
    write_latex_block(inference)

    summary = {
        "n_years": int(len(annual)),
        "best_models": best_model_name,
        "targets": {
            tgt: {
                "best_rmse": float(metrics[(metrics["target"] == tgt)].sort_values("rmse").iloc[0]["rmse"]),
                "best_r2": float(metrics[(metrics["target"] == tgt)].sort_values("rmse").iloc[0]["r2"]),
            }
            for tgt in ["inflation_rate_annual", "central_bank_interest_rate_annual"]
        },
    }
    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Generated outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
