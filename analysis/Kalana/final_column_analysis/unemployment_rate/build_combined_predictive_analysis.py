from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[4]
KALANA_DIR = Path(__file__).resolve().parents[2]
UNEMP_DIR = Path(__file__).resolve().parent
EMP_DIR = KALANA_DIR / "final_column_analysis" / "employment_ratio"
MASTER_CSV = REPO_ROOT / "Dataset_Management" / "SriLanka_Migration_final.csv"
UNEMP_MONTHLY = UNEMP_DIR / "unemployment_rate_monthly_1994_2025.csv"
EMP_MONTHLY = EMP_DIR / "employment_ratio_monthly_1994_2025.csv"

OUT_DIR = UNEMP_DIR / "combined_predictive_analysis_outputs"
FIG_DIR = OUT_DIR / "figures"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master = pd.read_csv(MASTER_CSV)
    master["date"] = pd.to_datetime(master["date"])

    cols = [
        "slbfe_total_annual",
        "remittances_annual_usd_mn",
        "employment_ratio_annual",
        "unemployment_rate_annual",
    ]
    for col in cols:
        master[col] = pd.to_numeric(master[col], errors="coerce")

    yearly = (
        master.assign(year=master["date"].dt.year)
        .groupby("year", as_index=False)[cols]
        .mean()
        .sort_values("year")
    )

    unemp_monthly = pd.read_csv(UNEMP_MONTHLY)
    unemp_monthly["date"] = pd.to_datetime(unemp_monthly["date"])

    emp_monthly = pd.read_csv(EMP_MONTHLY)
    emp_monthly["date"] = pd.to_datetime(emp_monthly["date"])

    return yearly, unemp_monthly, emp_monthly


def topic_3_1_eda(yearly: pd.DataFrame, unemp_monthly: pd.DataFrame, emp_monthly: pd.DataFrame) -> None:
    merged_monthly = pd.merge(
        unemp_monthly[["date", "unemployment_rate_monthly"]],
        emp_monthly[["date", "employment_ratio_monthly"]],
        on="date",
        how="inner",
    ).sort_values("date")

    corr_df = yearly[
        [
            "slbfe_total_annual",
            "remittances_annual_usd_mn",
            "employment_ratio_annual",
            "unemployment_rate_annual",
        ]
    ].corr()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.set_style("whitegrid")

    axes[0, 0].plot(
        merged_monthly["date"],
        merged_monthly["unemployment_rate_monthly"],
        color="#c0392b",
        linewidth=2,
    )
    axes[0, 0].set_title("Unemployment Rate (Monthly, 1994-2025)")
    axes[0, 0].set_ylabel("Unemployment Rate (%)")

    axes[0, 1].plot(
        merged_monthly["date"],
        merged_monthly["employment_ratio_monthly"],
        color="#1f618d",
        linewidth=2,
    )
    axes[0, 1].set_title("Employment Ratio (Monthly, 1994-2025)")
    axes[0, 1].set_ylabel("Employment Ratio")

    sns.heatmap(corr_df, cmap="RdYlGn", annot=True, fmt=".2f", ax=axes[1, 0], vmin=-1, vmax=1)
    axes[1, 0].set_title("Yearly Correlation Matrix")

    sns.scatterplot(
        data=yearly,
        x="remittances_annual_usd_mn",
        y="unemployment_rate_annual",
        ax=axes[1, 1],
        color="#8e44ad",
        s=70,
    )
    sns.regplot(
        data=yearly,
        x="remittances_annual_usd_mn",
        y="unemployment_rate_annual",
        ax=axes[1, 1],
        scatter=False,
        color="#2c3e50",
        line_kws={"linewidth": 2},
    )
    axes[1, 1].set_title("Remittances vs Unemployment (Yearly)")
    axes[1, 1].set_xlabel("Remittances Annual (USD Million)")
    axes[1, 1].set_ylabel("Unemployment Rate (%)")

    fig.suptitle("Topic 3.1 Exploratory Data Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "topic_3_1_eda_overview.png", dpi=220)
    plt.close(fig)


def topic_3_3_1_task_formulation() -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")

    boxes = [
        (0.04, 0.30, 0.22, 0.40, "Inputs\nslbfe_total_annual\nremittances_annual_usd_mn"),
        (0.33, 0.30, 0.22, 0.40, "Task Type\nSupervised Regression\n(Time-aware yearly data)"),
        (0.62, 0.55, 0.32, 0.25, "Target A\nunemployment_rate_annual"),
        (0.62, 0.18, 0.32, 0.25, "Target B\nemployment_ratio_annual"),
    ]

    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#ecf0f1", edgecolor="#2c3e50", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    ax.annotate("", xy=(0.33, 0.50), xytext=(0.26, 0.50), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.62, 0.67), xytext=(0.55, 0.50), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.62, 0.30), xytext=(0.55, 0.50), arrowprops=dict(arrowstyle="->", lw=2))

    ax.set_title("Topic 3.3.1 Machine Learning Task Formulation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "topic_3_3_1_task_formulation.png", dpi=220)
    plt.close(fig)


def time_series_cv_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_models(yearly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = yearly[["slbfe_total_annual", "remittances_annual_usd_mn"]].values
    targets = {
        "unemployment_rate_annual": yearly["unemployment_rate_annual"].values,
        "employment_ratio_annual": yearly["employment_ratio_annual"].values,
    }

    model_bank = {
        "LinearRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, max_depth=4),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    metric_rows: list[dict] = []
    pred_rows: list[dict] = []

    for target_name, y in targets.items():
        for model_name, model in model_bank.items():
            fold_rmse: list[float] = []
            fold_mae: list[float] = []
            fold_r2: list[float] = []

            for fold_id, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                rmse = time_series_cv_rmse(y_test, y_pred)
                mae = float(mean_absolute_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))

                fold_rmse.append(rmse)
                fold_mae.append(mae)
                fold_r2.append(r2)

                for i, idx in enumerate(test_idx):
                    pred_rows.append(
                        {
                            "target": target_name,
                            "model": model_name,
                            "fold": fold_id,
                            "year": int(yearly.iloc[idx]["year"]),
                            "actual": float(y_test[i]),
                            "predicted": float(y_pred[i]),
                        }
                    )

            metric_rows.append(
                {
                    "target": target_name,
                    "model": model_name,
                    "cv_rmse_mean": np.mean(fold_rmse),
                    "cv_rmse_std": np.std(fold_rmse, ddof=1),
                    "cv_mae_mean": np.mean(fold_mae),
                    "cv_mae_std": np.std(fold_mae, ddof=1),
                    "cv_r2_mean": np.mean(fold_r2),
                    "cv_r2_std": np.std(fold_r2, ddof=1),
                }
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["target", "cv_rmse_mean"])
    preds_df = pd.DataFrame(pred_rows)
    return metrics_df, preds_df


def topic_3_3_2_validation_strategy() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_title("Topic 3.3.2 Evaluation Metrics & Validation Strategy", fontsize=14, fontweight="bold")

    n_samples = 32
    years = np.arange(1994, 2026)
    tscv = TimeSeriesSplit(n_splits=5)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(np.arange(n_samples)), start=1):
        y = np.full(n_samples, fold)
        ax.scatter(years[train_idx], y[train_idx], marker="s", s=100, color="#2ecc71", label="Train" if fold == 1 else "")
        ax.scatter(years[test_idx], y[test_idx], marker="s", s=100, color="#e74c3c", label="Test" if fold == 1 else "")

    ax.set_xlabel("Year")
    ax.set_ylabel("Fold")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    ax.text(
        1994,
        0.25,
        "Metrics: RMSE, MAE, R^2 | Expanding-window TimeSeriesSplit (5 folds)",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "topic_3_3_2_validation_strategy.png", dpi=220)
    plt.close(fig)


def topic_3_3_3_ml_results(metrics_df: pd.DataFrame, preds_df: pd.DataFrame) -> None:
    best = metrics_df.sort_values(["target", "cv_rmse_mean"]).groupby("target", as_index=False).first()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.barplot(
        data=metrics_df,
        x="model",
        y="cv_rmse_mean",
        hue="target",
        ax=axes[0, 0],
        palette="Set2",
    )
    axes[0, 0].set_title("Cross-Validated RMSE")
    axes[0, 0].set_ylabel("RMSE")

    sns.barplot(
        data=metrics_df,
        x="model",
        y="cv_r2_mean",
        hue="target",
        ax=axes[0, 1],
        palette="Set1",
    )
    axes[0, 1].set_title("Cross-Validated R^2")
    axes[0, 1].set_ylabel("R^2")

    for i, target in enumerate(["unemployment_rate_annual", "employment_ratio_annual"]):
        chosen_model = best.loc[best["target"] == target, "model"].iloc[0]
        temp = preds_df[(preds_df["target"] == target) & (preds_df["model"] == chosen_model)].sort_values("year")
        row = 1
        col = i
        axes[row, col].plot(temp["year"], temp["actual"], color="#34495e", linewidth=2, label="Actual")
        axes[row, col].plot(temp["year"], temp["predicted"], color="#f39c12", linewidth=2, linestyle="--", label="Predicted")
        axes[row, col].set_title(f"Best Model ({chosen_model}) - {target}")
        axes[row, col].set_xlabel("Year")
        axes[row, col].legend()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles[:2], ["unemployment_rate_annual", "employment_ratio_annual"], loc="upper right")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        axes[0, 1].legend(handles[:2], ["unemployment_rate_annual", "employment_ratio_annual"], loc="lower right")

    fig.suptitle("Topic 3.3.3 Machine Learning Results", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "topic_3_3_3_ml_results.png", dpi=220)
    plt.close(fig)


def run_hypothesis_tests(yearly: pd.DataFrame) -> pd.DataFrame:
    tests = [
        {
            "hypothesis": "H1",
            "x": "remittances_annual_usd_mn",
            "y": "unemployment_rate_annual",
            "alternative": "negative",
        },
        {
            "hypothesis": "H2",
            "x": "slbfe_total_annual",
            "y": "employment_ratio_annual",
            "alternative": "positive",
        },
    ]

    rows: list[dict] = []
    for t in tests:
        x = yearly[t["x"]].values
        y = yearly[t["y"]].values

        shapiro_x = stats.shapiro(x)
        shapiro_y = stats.shapiro(y)
        spearman = stats.spearmanr(x, y)

        if t["alternative"] == "negative":
            p_one = spearman.pvalue / 2 if spearman.statistic < 0 else 1 - (spearman.pvalue / 2)
        else:
            p_one = spearman.pvalue / 2 if spearman.statistic > 0 else 1 - (spearman.pvalue / 2)

        rows.append(
            {
                "hypothesis": t["hypothesis"],
                "x_variable": t["x"],
                "y_variable": t["y"],
                "alternative": t["alternative"],
                "n_years": len(yearly),
                "shapiro_x_W": shapiro_x.statistic,
                "shapiro_x_p": shapiro_x.pvalue,
                "shapiro_y_W": shapiro_y.statistic,
                "shapiro_y_p": shapiro_y.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p_two_sided": spearman.pvalue,
                "spearman_p_one_sided": p_one,
            }
        )

    return pd.DataFrame(rows)


def topic_3_3_4_statistical_inference(yearly: pd.DataFrame, inf_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    h1 = inf_df.loc[inf_df["hypothesis"] == "H1"].iloc[0]
    h2 = inf_df.loc[inf_df["hypothesis"] == "H2"].iloc[0]

    sns.regplot(
        data=yearly,
        x="remittances_annual_usd_mn",
        y="unemployment_rate_annual",
        ax=axes[0],
        scatter_kws={"s": 55, "color": "#8e44ad"},
        line_kws={"color": "#2c3e50", "linewidth": 2},
    )
    axes[0].set_title("H1: Remittances vs Unemployment")
    axes[0].set_xlabel("Remittances Annual (USD Million)")
    axes[0].set_ylabel("Unemployment Rate (%)")
    axes[0].text(
        0.03,
        0.95,
        f"Spearman rho = {h1['spearman_rho']:.3f}\np(one-sided) = {h1['spearman_p_one_sided']:.4g}",
        transform=axes[0].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#555"},
    )

    sns.regplot(
        data=yearly,
        x="slbfe_total_annual",
        y="employment_ratio_annual",
        ax=axes[1],
        scatter_kws={"s": 55, "color": "#16a085"},
        line_kws={"color": "#1c2833", "linewidth": 2},
    )
    axes[1].set_title("H2: Emigration vs Employment Ratio")
    axes[1].set_xlabel("SLBFE Total Annual Emigration")
    axes[1].set_ylabel("Employment Ratio")
    axes[1].text(
        0.03,
        0.95,
        f"Spearman rho = {h2['spearman_rho']:.3f}\np(one-sided) = {h2['spearman_p_one_sided']:.4g}",
        transform=axes[1].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#555"},
    )

    fig.suptitle("Topic 3.3.4 Statistical Inference", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "topic_3_3_4_statistical_inference.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    yearly, unemp_monthly, emp_monthly = load_data()

    yearly.to_csv(OUT_DIR / "combined_yearly_dataset_1994_2025.csv", index=False)

    topic_3_1_eda(yearly, unemp_monthly, emp_monthly)
    topic_3_3_1_task_formulation()
    topic_3_3_2_validation_strategy()

    metrics_df, preds_df = evaluate_models(yearly)
    metrics_df.to_csv(OUT_DIR / "topic_3_3_3_model_metrics.csv", index=False)
    preds_df.to_csv(OUT_DIR / "topic_3_3_3_predictions_by_fold.csv", index=False)
    topic_3_3_3_ml_results(metrics_df, preds_df)

    inf_df = run_hypothesis_tests(yearly)
    inf_df.to_csv(OUT_DIR / "topic_3_3_4_hypothesis_tests.csv", index=False)
    topic_3_3_4_statistical_inference(yearly, inf_df)

    print("Analysis completed.")
    print(f"Outputs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()