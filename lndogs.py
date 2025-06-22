"""
Grain-Size Distribution Deconvolution Tool.

This script computes potential deconvolutions of a grain-size distribution
by fitting a combination of three weighted normal distributions.
"""

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import leastsq

from helpers import (
    compute_errors,
    differentiate,
    gauss,
    normalize,
    phi,
    triple_phi,
)

# Constants for plotting
LINEWIDTH = 7
MARKERSIZE = 10
MARKERWIDTH = 5
LEGEND_LOCATION = "upper left"
NUM_INTERP_POINTS = 500


def load_data(filepath: str, col: int = 1):
    """Load and preprocess grain-size distribution data from an Excel file.

    Args:
        filepath: Path to the Excel file containing the data.
        col: Zero-based index of the column to process (default: 1).

    Returns:
        A tuple containing:
        - params0: Initial guess parameters for optimization
        - datax: Original x-axis data (grain size in phi scale)
        - datay: Original y-axis data (cumulative distribution)
        - distribx: Differentiated x-axis data
        - distriby: Differentiated y-axis data
        - interp_datax: Interpolated x-axis data
        - interp_datay: Interpolated y-axis data
    """
    df = pd.read_excel(filepath)
    nparray = df.to_numpy()
    datax = nparray[:, 0]
    datay = nparray[:, col]

    minx = min(datax)
    maxx = max(datax)

    distribx, distriby = differentiate(datax, datay)

    interp_datax = np.arange(minx, maxx, (maxx - minx) / NUM_INTERP_POINTS)
    interp_datay = np.interp(interp_datax, datax, datay)

    # Calculate initial parameter estimates
    middle_mu = np.interp(50, datay, datax)
    min_mu = np.interp(20, datay, datax)
    max_mu = np.interp(80, datay, datax)
    bs = (maxx - minx) / 6

    params0 = np.array([
        middle_mu, bs, 75,  # First distribution parameters
        min_mu, bs - 0.2, 15,  # Second distribution parameters
        max_mu, bs + 0.2, 10,  # Third distribution parameters
    ])

    return (
        params0,
        datax,
        datay,
        distribx,
        distriby,
        interp_datax,
        interp_datay,
    )


def optimize(
    params0: np.ndarray,
    interp_datay: np.ndarray,
    interp_datax: np.ndarray,
    max_iterations: int = 100000,
) -> np.ndarray:
    """Optimize the distribution parameters using least squares fitting.

    Args:
        params0: Initial parameter guesses
        interp_datay: Interpolated y-axis data for fitting
        interp_datax: Interpolated x-axis data for fitting
        max_iterations: Maximum number of iterations for optimization

    Returns:
        Optimized parameters
    """
    result = leastsq(
        compute_errors,
        params0,
        args=(interp_datay, interp_datax),
        full_output=True,
        maxfev=max_iterations,
    )
    return result[0]


def print_results(best_params: np.ndarray) -> None:
    """Print the optimized parameters in a readable format.

    Args:
        best_params: Optimized parameters from the fitting process
    """
    print("Best parameters:")
    print("======================")
    print(f"mu1    : {best_params[0]:.2f}")
    print(f"sigma1 : {abs(best_params[1]):.2f}")
    print(f"weight1: {normalize(best_params[2], best_params[5], best_params[8]):.2f}")
    print(f"mu2    : {best_params[3]:.2f}")
    print(f"sigma2 : {abs(best_params[4]):.2f}")
    print(f"weight2: {normalize(best_params[5], best_params[2], best_params[8]):.2f}")
    print(f"mu3    : {best_params[6]:.2f}")
    print(f"sigma3 : {abs(best_params[7]):.2f}")
    print(f"weight3: {normalize(best_params[8], best_params[2], best_params[5]):.2f}")
    print("------------------------------------")


def draw_results(
    datax,
    datay,
    distribx,
    distriby,
    interp_datax,
    best_params,
) -> None:
    """Visualize the original data and fitted distributions.

    Args:
        datax: Original x-axis data
        datay: Original y-axis data
        distribx: Differentiated x-axis data
        distriby: Differentiated y-axis data
        interp_datax: Interpolated x-axis data
        best_params: Optimized distribution parameters
    """
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)  # Cumulative functions
    ax2 = fig.add_subplot(122)  # Distributions

    # Prepare data for plotting
    datafit = triple_phi(interp_datax, best_params)
    fun1 = 100 * phi(interp_datax, best_params[0], best_params[1])

    # Plot cumulative functions
    ax1.plot(
        datax,
        datay,
        "x",
        markersize=MARKERSIZE,
        markeredgewidth=MARKERWIDTH,
        label="data",
    )
    ax1.plot(
        interp_datax,
        datafit,
        "r",
        linewidth=LINEWIDTH,
        label="fit",
    )

    # Add individual distribution components
    for i, (color, offset) in enumerate(zip(["b", "g", "y"], [0, 3, 6])):
        mu = best_params[offset]
        sigma = abs(best_params[offset + 1])
        weight = normalize(
            best_params[offset + 2],
            best_params[(offset + 5) % 9],
            best_params[(offset + 8) % 9],
        )
        label = f"g{i+1} ({mu:.2f}/{sigma:.2f}/{weight:.2f})"
        ax1.plot(
            interp_datax,
            100 * phi(interp_datax, mu, sigma),
            color,
            linewidth=LINEWIDTH,
            label=label,
        )

    ax1.legend(loc=LEGEND_LOCATION)
    ax1.set_title("Cumulative functions")
    ax1.grid(True)

    # Plot distributions
    fitx, fity = differentiate(interp_datax, datafit)
    x1, y1 = differentiate(interp_datax, fun1)

    ax2.plot(
        distribx,
        distriby,
        "x",
        markersize=MARKERSIZE,
        markeredgewidth=MARKERWIDTH,
        label="data",
    )
    ax2.plot(
        fitx,
        fity,
        "r",
        linewidth=LINEWIDTH,
        label="fit",
    )

    # Add individual distribution components
    for i, (color, offset) in enumerate(zip(["b", "g", "y"], [0, 3, 6])):
        ax2.plot(
            x1,
            100 * gauss(x1, best_params[offset], best_params[offset + 1]),
            color,
            linewidth=LINEWIDTH,
            label=f"g{i+1}",
        )

    ax2.legend(loc=LEGEND_LOCATION)
    ax2.set_title("Distributions")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main(filepath: str = "./sample.xlsx", column: int = 1, draw: bool = True) -> None:
    """Grain-Size Distribution Deconvolution Tool.

    This script computes potential deconvolutions of a grain-size distribution by fitting
    a combination of three weighted normal distributions.

    Data Source Requirements:
    -------------------------
    The input data must be provided in an Excel (`.xlsx`) file with the following structure:

    - **Header Row**: The first row contains headers and is ignored during data import.
    - **Grain Size (phi scale)**: The first column (Python index `0`) should contain
      grain-size values in the phi scale.
    - **Cumulative Distribution Data**: Subsequent columns contain the cumulative
      distribution values for each sample.

    Usage Notes:
    -----------
    - By default, the script processes the **second column** (Python index `1`) for deconvolution.
    - The tool analyzes **one column at a time**, allowing for targeted distribution fitting.

    Custom Column Selection:
    -----------------------
    To process a specific column, simply specify its zero-based index
    (e.g., 0 for the first column, 1 for the second, etc.).

    Example:
    --------
    By default, column 1 (the second column) is used.
    To analyze a different column, provide its index when running the script.

    Args:
        filepath: Path to the Excel file containing the data.
        column: Zero-based index of the column to process (default: 1).
        draw: Whether to display the visualization (default: True).
    """
    # Load data
    params0, datax, datay, distribx, distriby, interp_datax, interp_datay = load_data(
        filepath, column
    )

    # Optimize (curve fitting)
    best_params = optimize(params0, interp_datay, interp_datax)

    # Print results
    print_results(best_params)

    # Draw chart if requested
    if draw:
        draw_results(datax, datay, distribx, distriby, interp_datax, best_params)


if __name__ == "__main__":
    fire.Fire(main)
