"""Small statistical helpers shared by the dataset notebooks."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint

AGE_BANDS = [0, 30, 45, 60, 75, 120]
AGE_LABELS = ["< 30", "30-44", "45-59", "60-74", "75+"]


def age_band(age: pd.Series) -> pd.Series:
    """
    Group ages into the bands used across all notebooks.

    Parameters
    ----------
    age : pd.Series
        Age in years; NaN stays NaN.

    Returns
    -------
    pd.Series
        Ordered categorical band.
    """
    return pd.cut(age, AGE_BANDS, labels=AGE_LABELS, right=False)


def prevalence(frame: pd.DataFrame, group: str, outcomes: list[str]) -> pd.DataFrame:
    """
    Share of records with each outcome per group, with 95% Wilson intervals.

    Parameters
    ----------
    frame : pd.DataFrame
        One row per record, with boolean or 0/1 outcome columns. Records where
        an outcome is NaN (not annotated) are left out for that outcome.
    group : str
        Grouping column.
    outcomes : list[str]
        Outcome columns.

    Returns
    -------
    pd.DataFrame
        Long table with ``group``, ``outcome``, ``records``, ``cases``,
        ``prevalence``, ``lower`` and ``upper``.
    """
    rows = []
    for value, subset in frame.groupby(group, observed=True):
        for outcome in outcomes:
            labeled = subset[outcome].dropna()
            if labeled.empty:
                continue
            cases = int(labeled.sum())
            lower, upper = proportion_confint(cases, len(labeled), method="wilson")
            rows.append({group: value, "outcome": outcome, "records": len(labeled), "cases": cases,
                         "prevalence": cases / len(labeled), "lower": lower, "upper": upper})
    table = pd.DataFrame(rows)
    if isinstance(frame[group].dtype, pd.CategoricalDtype):
        table[group] = pd.Categorical(table[group], categories=frame[group].cat.categories, ordered=True)
    return table.sort_values([group, "outcome"], kind="stable").reset_index(drop=True)


def association_test(frame: pd.DataFrame, group: str, outcome: str) -> float:
    """
    Chi-square test of independence between a grouping and an outcome.

    Parameters
    ----------
    frame : pd.DataFrame
        One row per record.
    group : str
        Categorical column.
    outcome : str
        Boolean column.

    Returns
    -------
    float
        p-value.
    """
    table = pd.crosstab(frame[group], frame[outcome])
    return float(chi2_contingency(table)[1])


def odds_ratios(frame: pd.DataFrame, outcomes: list[str], age: str = "age",
                male: str = "male") -> pd.DataFrame:
    """
    Odds ratio of each outcome per 10 years of age and for male sex.

    Each outcome gets its own logistic regression with age (per decade) and
    male sex as predictors, fitted on records where both are known.

    Parameters
    ----------
    frame : pd.DataFrame
        One row per record with numeric age, boolean sex and boolean outcomes.
    outcomes : list[str]
        Boolean outcome columns.
    age : str
        Age column in years.
    male : str
        Boolean column, True for male.

    Returns
    -------
    pd.DataFrame
        One row per outcome and predictor with the odds ratio, its 95%
        interval and p-value.
    """
    rows = []
    for outcome in outcomes:
        known = frame.dropna(subset=[age, male, outcome])
        predictors = pd.DataFrame({"age per 10 years": known[age] / 10, "male": known[male].astype(float)})
        fit = sm.Logit(known[outcome].astype(float), sm.add_constant(predictors)).fit(disp=False)
        intervals = np.exp(fit.conf_int())
        for name in ("age per 10 years", "male"):
            rows.append({"outcome": outcome, "predictor": name, "odds_ratio": np.exp(fit.params[name]),
                         "lower": intervals.loc[name, 0], "upper": intervals.loc[name, 1],
                         "p_value": fit.pvalues[name]})
    return pd.DataFrame(rows)


def plot_prevalence(table: pd.DataFrame, group: str, axis: object, title: str, connect: bool = True) -> None:
    """
    Plot prevalence per group with interval error bars, one series per outcome.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``prevalence``.
    group : str
        Grouping column used in ``prevalence``.
    axis : object
        Matplotlib axis.
    title : str
        Axis title.
    connect : bool
        Join the points with lines. Use False for unordered groups such as sex,
        where the series are drawn side by side instead.
    """
    outcomes = list(dict.fromkeys(table["outcome"]))
    width = 0 if connect else 0.6 / len(outcomes)
    for number, outcome in enumerate(outcomes):
        rows = table[table["outcome"] == outcome]
        positions = np.arange(len(rows)) + (number - (len(outcomes) - 1) / 2) * width
        errors = [rows["prevalence"] - rows["lower"], rows["upper"] - rows["prevalence"]]
        axis.errorbar(positions, rows["prevalence"], yerr=errors, marker="o", capsize=3, label=outcome,
                      linestyle="-" if connect else "none")
    axis.set_xticks(np.arange(len(rows)), rows[group].astype(str))
    axis.set(title=title, ylabel="Share of records")
    axis.legend(fontsize=8)
