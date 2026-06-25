"""likelihood_crb.py — log-likelihood profile + Cramér–Rao bound.

Simulates N Gaussian measurements, plots the log-likelihood as a function
of the mean parameter mu, and overlays the Cramér–Rao bound (variance floor).

Output: figures/likelihood_crb.{pdf,png}
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib

rng = np.random.default_rng(seed=42)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.6,
})

# ── True parameters ────────────────────────────────────────────────────────
mu_true = 5.0
sigma_known = 1.0      # known noise; realistic for a calibrated instrument
N = 10

# ── Simulated data ─────────────────────────────────────────────────────────
data = rng.normal(mu_true, sigma_known, N)
mu_hat = data.mean()   # MLE for Gaussian, known sigma

# ── Log-likelihood profile ─────────────────────────────────────────────────
mu_grid = np.linspace(2.5, 7.5, 500)
log_L = np.array([
    -0.5 * np.sum(((data - mu) / sigma_known)**2)
    for mu in mu_grid
])
log_L -= log_L.max()   # normalise to peak = 0

# Cramér–Rao bound: Var(mu_hat) >= 1/I(mu) = sigma^2/N
crb_variance = sigma_known**2 / N
crb_sigma = np.sqrt(crb_variance)

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 3.8))

ax.plot(mu_grid, log_L, color="#1a4a7a", label=r"$\ln \mathcal{L}(\mu;\,\mathbf{x})$  (normalised)")

# −1/2 level → 1-sigma confidence interval from likelihood
ax.axhline(-0.5, color="#1a4a7a", linewidth=0.8, linestyle="--", alpha=0.6)
ax.text(mu_grid[-1] * 0.98, -0.5 + 0.04, r"$\Delta\ln\mathcal{L} = -1/2$",
        ha="right", va="bottom", fontsize=8, color="#555")

# MLE vertical
ax.axvline(mu_hat, color="#c75000", linewidth=1.2, linestyle=":",
           label=fr"MLE $\hat{{\mu}} = {mu_hat:.2f}$")

# CRB shading
ax.axvspan(mu_hat - crb_sigma, mu_hat + crb_sigma, color="#2e7d32",
           alpha=0.10, label=fr"CRB: $\sigma(\hat{{\mu}}) \geq {crb_sigma:.3f}$")
ax.axvline(mu_hat - crb_sigma, color="#2e7d32", linewidth=0.8, linestyle="--")
ax.axvline(mu_hat + crb_sigma, color="#2e7d32", linewidth=0.8, linestyle="--")

# True value
ax.axvline(mu_true, color="#888", linewidth=0.9, linestyle="-.",
           label=fr"True $\mu_0 = {mu_true}$")

ax.set_xlabel(r"Parameter $\mu$")
ax.set_ylabel(r"Normalised log-likelihood")
ax.set_ylim(-3.5, 0.3)
ax.legend(frameon=False, fontsize=8)
ax.set_title(
    fr"$N={N}$ measurements, $\sigma={sigma_known}$  —  CRB variance $= \sigma^2/N = {crb_variance:.2f}$",
    fontsize=9
)

repo_root = pathlib.Path(__file__).resolve().parents[2]
out = repo_root / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "likelihood_crb.pdf", bbox_inches="tight")
fig.savefig(out / "likelihood_crb.png", bbox_inches="tight", dpi=150)
plt.close(fig)

if __name__ == "__main__":
    print(f"MLE: {mu_hat:.4f},  CRB sigma: {crb_sigma:.4f}")
