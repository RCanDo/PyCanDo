# %% imports
import warnings
warnings.filterwarnings('ignore')

from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_calibration(
    y: pd.Series,
    y_fit: pd.Series,
    y_calib: pd.Series = None,
    title: str = None,
    prob_true_fit: np.ndarray = None,
    prob_fit: np.ndarray = None,
    prob_true_calib: np.ndarray = None,
    prob_calib: np.ndarray = None,
    strategy: str = "quantile",
    n_bins: int = 10,
) -> None:
    """
    Plot calibration curve for one or eventually two models (i.e. before and after calibration)

    """
    plt.figure(figsize=(10, 10))

    if prob_fit is None and prob_true_fit is None:
        prob_true_fit, prob_fit = calibration_curve(y, y_fit, strategy=strategy, n_bins=n_bins)
    if y_calib is not None or prob_calib is not None:
        if prob_true_calib is None and prob_calib is None:
            prob_true_calib, prob_calib = calibration_curve(y, y_calib, strategy=strategy, n_bins=n_bins)
        plt.plot(prob_calib, prob_true_calib, linewidth=2, marker='o')
    plt.plot(prob_fit, prob_true_fit, linewidth=2, marker='o')
    plt.xlabel('predicted prob', fontsize=14)
    plt.ylabel('empirical prob', fontsize=14)

    axis_lim_fit = max(max(prob_fit), max(prob_true_fit))
    if y_calib is not None or prob_calib is not None:
        axis_lim = max(max(prob_calib), max(prob_true_calib), axis_lim_fit)
    axis_lim = axis_lim * 1.1
    try:
        plt.ylim((0, axis_lim))
        plt.xlim((0, axis_lim))
    except Exception as e:
        plt.ylim((0, axis_lim_fit))
        plt.xlim((0, axis_lim_fit))
        print(e)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    names_legend = ['fit', 'ideal'] if y_calib is None and prob_calib is None else ['calib', 'fit', 'ideal']
    plt.legend(names_legend, fontsize=10)
    if title is not None:
        plt.title(title)
    plt.show()


def cr_on_intervals(
        quantiles: list | np.ndarray,
        df: pd.DataFrame,
) -> Tuple[list, pd.DataFrame]:
    """
    For a given list of quantiles of probabilities the true conversion rate
    and number of ones in the target is calculated in each interval between quantiles.
    """
    q = sorted(list(set(quantiles)))
    df_temp = df[df["probs"] <= q[0]]
    values = {q[0] / 2: df_temp["y"].sum() / len(df_temp)}
    df_q = pd.DataFrame(
        [[f"(0, {q[0]}]", round(df_temp["probs"].mean(), 8), df_temp["y"].sum(), len(df_temp)]],
        columns=["interval", "probs_mean", "1's", 'nobs'])
    for i in range(len(q) - 1):
        df_temp = df[(q[i] < df["probs"]) & (df["probs"] <= q[i + 1])]
        values[(q[i] + q[i + 1]) / 2] = df_temp["y"].sum() / len(df_temp)
        row = [f"({q[i]}, {q[i + 1]}]", round(df_temp["probs"].mean(), 8), df_temp["y"].sum(), len(df_temp)]
        df_q.loc[len(df_q)] = row
    df_temp = df[df["probs"] > q[-1]]
    values[(q[-1] + max(df['probs'])) / 2] = df_temp["y"].sum() / len(df_temp)
    row = [f"({q[-1]}, {max(df['probs'])}]", round(df_temp["probs"].mean(), 8), df_temp["y"].sum(), len(df_temp)]
    df_q.loc[len(df_q)] = row
    df_q['cr'] = round(df_q["1's"] / df_q["nobs"], 8)
    return values, df_q


def concat(
        y: np.ndarray,
        y_probs: np.ndarray
) -> pd.DataFrame:
    """
    Concatenate series of probabilities and series of true values and into a DataFrame
    """
    df = pd.DataFrame(y_probs, columns=["probs"])
    df.index = y.index
    df = pd.concat([df, y], axis=1, ignore_index=True)
    df.columns = ["probs", "y"]
    return df


def plot_calibration_2(
    y: pd.Series,
    y_fit: pd.Series,
    y_calib: pd.Series = None,
    title: str = None,
    mean: bool = True,
) -> None:
    """
    Plot calibration curves but without using functions from external library.
    """
    plt.figure(figsize=(10, 10))

    df_fit = concat(y, y_fit)
    df_calib = concat(y, y_calib)

    q_fit = np.quantile(y_fit, np.arange(0.1, 1, 0.1))
    q_calib = np.quantile(y_calib, np.arange(0.1, 1, 0.1))

    fit_values, df_q_fit = cr_on_intervals(q_fit, df_fit)
    calib_values, df_q_calib = cr_on_intervals(q_calib, df_calib)

    print("Fit")
    print(df_q_fit)
    print("\n")
    print("Calib")
    print(df_q_calib)

    x_fit = df_q_fit["probs_mean"] if mean else list(fit_values.keys())
    y_fit = list(fit_values.values())
    x_calib = df_q_calib["probs_mean"] if mean else list(calib_values.keys())
    y_calib = list(calib_values.values())

    plt.plot(x_calib, y_calib, linewidth=2, marker='o')
    plt.plot(x_fit, y_fit, linewidth=2, marker='o')
    plt.xlabel('predicted prob', fontsize=14)
    plt.ylabel('empirical prob', fontsize=14)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    axis_lim = max(
        max(x_fit),
        max(y_fit),
        max(x_calib),
        max(y_calib),
    ) * 1.1
    plt.ylim((0, axis_lim))
    plt.xlim((0, axis_lim))
    names_legend = ['calib', 'fit', 'ideal']
    plt.legend(names_legend, fontsize=10)
    if title is not None:
        plt.title(title)
    plt.show()
