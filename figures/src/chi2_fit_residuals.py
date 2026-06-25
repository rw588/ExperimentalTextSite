"""chi2_fit_residuals.py — weighted least-squares fit with residual panel.

Reads data/sample_calibration.csv, fits y = a*x + b by weighted least squares,
plots the fit and residuals, and annotates chi-squared / dof and parameter
values with uncertainties.

Run standalone:   python figures/src/chi2_fit_residuals.py
In Quarto:        called via the code cell in 03-estimation-fisher.qmd
Output:           figures/chi2_fit_residuals.pdf  (gitignored)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless builds
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2 as chi2_dist

# ── Reproducibility ────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Load data ──────────────────────────────────────────────────────────────
import pathlib
repo_root = pathlib.Path(__file__).resolve().parents[2]
df = pd.read_csv(repo_root / "data" / "sample_calibration.csv")
x, y, sigma = df["x"].values, df["y"].values, df["sigma_y"].values
w = 1.0 / sigma**2                  # weights

# ── Weighted least-squares: y = a*x + b ───────────────────────────────────
# Design matrix
A = np.column_stack([x, np.ones_like(x)])
W = np.diag(w)
# Normal equations:  (A^T W A) θ = A^T W y
ATA = A.T @ W @ A
ATy = A.T @ (W @ y)
cov = np.linalg.inv(ATA)
theta = cov @ ATy
a_hat, b_hat = theta
sigma_a = np.sqrt(cov[0, 0])
sigma_b = np.sqrt(cov[1, 1])

# ── Goodness of fit ────────────────────────────────────────────────────────
y_fit = a_hat * x + b_hat
residuals = y - y_fit
chi2_val = float(np.sum((residuals / sigma) ** 2))
ndof = len(x) - 2
chi2_red = chi2_val / ndof
p_val = 1.0 - chi2_dist.cdf(chi2_val, ndof)

# ── Plot ───────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(6.0, 5.5),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    sharex=True,
)

# Top: data + fit
ax1.errorbar(x, y, yerr=sigma, fmt="o", color="#1a4a7a",
             markersize=4, linewidth=0.8, capsize=3, label="Data")
x_line = np.linspace(x.min(), x.max(), 300)
ax1.plot(x_line, a_hat * x_line + b_hat, color="#c75000",
         linewidth=1.6, label=fr"Fit: $\hat{{a}}={a_hat:.4f}\pm{sigma_a:.4f}$"
                              fr",  $\hat{{b}}={b_hat:.4f}\pm{sigma_b:.4f}$")
ax1.set_ylabel(r"$y$  (arb. units)")
ax1.legend(loc="upper left", frameon=False)
ax1.set_title(
    fr"$\chi^2 = {chi2_val:.1f}$,  "
    fr"$\nu = {ndof}$,  "
    fr"$\chi^2_\nu = {chi2_red:.2f}$,  "
    fr"$p = {p_val:.2f}$",
    fontsize=9, color="#333"
)

# Bottom: normalised residuals
ax2.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
ax2.fill_between(x, -1, 1, color="#1a4a7a", alpha=0.08, label=r"$\pm 1\sigma$")
ax2.errorbar(x, residuals / sigma, yerr=1, fmt="o", color="#1a4a7a",
             markersize=4, linewidth=0.8, capsize=3)
ax2.set_ylabel(r"$(y - \hat{y})\,/\,\sigma$")
ax2.set_xlabel(r"$x$  (arb. units)")
ax2.set_ylim(-3.5, 3.5)
ax2.legend(loc="upper right", frameon=False, fontsize=8)

# ── Save ───────────────────────────────────────────────────────────────────
out_dir = repo_root / "figures"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "chi2_fit_residuals.pdf", bbox_inches="tight")
fig.savefig(out_dir / "chi2_fit_residuals.png", bbox_inches="tight", dpi=150)
plt.close(fig)

if __name__ == "__main__":
    print(f"Saved chi2_fit_residuals.{{pdf,png}}")
    print(f"a = {a_hat:.4f} ± {sigma_a:.4f}")
    print(f"b = {b_hat:.4f} ± {sigma_b:.4f}")
    print(f"χ²/ν = {chi2_red:.3f}  (p = {p_val:.3f})")
