#!/usr/bin/env python3
"""
Compare every @tangent.to/proba distribution against scipy.stats:
logpdf/logpmf, cdf, quantile (ppf), mean and variance, on grids spanning
the body and both tails, for two parameter sets each.

Run from the package root:

    uv run --with scipy python3 tests_compare-to-scipy/compare_with_scipy.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
NODE_SCRIPT = ROOT / "tests_compare-to-scipy" / "compare_proba.mjs"

# (proba name, params, scipy frozen distribution, discrete?)
CASES = [
    ("normal", {"mu": 0, "sigma": 1}, stats.norm(0, 1), False),
    ("normal", {"mu": -3.5, "sigma": 0.2}, stats.norm(-3.5, 0.2), False),
    ("uniform", {"low": -2, "high": 5}, stats.uniform(-2, 7), False),
    ("exponential", {"lambda": 1.7}, stats.expon(scale=1 / 1.7), False),
    ("lognormal", {"mu": 0.5, "sigma": 0.8}, stats.lognorm(0.8, scale=np.exp(0.5)), False),
    ("halfnormal", {"sigma": 2.0}, stats.halfnorm(scale=2.0), False),
    ("gamma", {"alpha": 2.5, "beta": 1.5}, stats.gamma(2.5, scale=1 / 1.5), False),
    ("gamma", {"alpha": 0.5, "beta": 3.0}, stats.gamma(0.5, scale=1 / 3.0), False),
    ("beta", {"alpha": 2, "beta": 5}, stats.beta(2, 5), False),
    ("beta", {"alpha": 0.5, "beta": 0.5}, stats.beta(0.5, 0.5), False),
    ("studentT", {"nu": 4, "mu": 1, "sigma": 2}, stats.t(4, loc=1, scale=2), False),
    ("studentT", {"nu": 30, "mu": 0, "sigma": 1}, stats.t(30), False),
    ("chi2", {"k": 3}, stats.chi2(3), False),
    ("chi2", {"k": 17}, stats.chi2(17), False),
    ("f", {"d1": 5, "d2": 12}, stats.f(5, 12), False),
    ("f", {"d1": 1, "d2": 4}, stats.f(1, 4), False),
    ("bernoulli", {"p": 0.3}, stats.bernoulli(0.3), True),
    ("binomial", {"n": 20, "p": 0.4}, stats.binom(20, 0.4), True),
    ("binomial", {"n": 500, "p": 0.02}, stats.binom(500, 0.02), True),
    ("poisson", {"lambda": 3.5}, stats.poisson(3.5), True),
    ("poisson", {"lambda": 80}, stats.poisson(80), True),
]

P_GRID = [1e-6, 0.001, 0.025, 0.2, 0.5, 0.8, 0.975, 0.999, 1 - 1e-6]

FAILURES = []


def check(label, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  ({detail})")
    if not ok:
        FAILURES.append(label)


def run_node(spec):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(spec, fh)
        path = fh.name
    r = subprocess.run(["node", str(NODE_SCRIPT), path], check=True,
                       capture_output=True, text=True, cwd=ROOT)
    return json.loads(r.stdout)


def main():
    print("scipy.stats comparison for @tangent.to/proba")
    for name, params, sp, discrete in CASES:
        label = f"{name} {params}"
        if discrete:
            qs = sp.ppf([0.001, 0.1, 0.5, 0.9, 0.999]).astype(int)
            xs = sorted(set(int(k) for k in qs))
            logpdf_sp = sp.logpmf(xs)
        else:
            xs = sp.ppf([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]).tolist()
            logpdf_sp = sp.logpdf(xs)

        js = run_node({"dist": name, "params": params, "x": list(map(float, xs)), "p": P_GRID})

        e_lp = float(np.max(np.abs(np.asarray(js["logpdf"]) - logpdf_sp)))
        e_cdf = float(np.max(np.abs(np.asarray(js["cdf"]) - sp.cdf(xs))))

        q_sp = sp.ppf(P_GRID)
        q_js = np.asarray(js["quantile"])
        if discrete:
            e_q = float(np.max(np.abs(q_js - q_sp)))  # integer match expected
        else:
            scale = np.maximum(1e-12, np.abs(q_sp))
            e_q = float(np.max(np.abs(q_js - q_sp) / scale))

        # moments: JSON serializes NaN as null; skip where scipy's moment is
        # non-finite (undefined-moment conventions differ: proba NaN, scipy inf)
        js_m = np.nan if js["mean"] is None else js["mean"]
        js_v = np.nan if js["variance"] is None else js["variance"]
        e_m = abs(js_m - sp.mean()) / max(1, abs(sp.mean())) if np.isfinite(sp.mean()) else 0.0
        e_v = abs(js_v - sp.var()) / max(1, abs(sp.var())) if np.isfinite(sp.var()) else 0.0

        ok = e_lp < 1e-9 and e_cdf < 1e-10 and e_q < (1e-7 if not discrete else 0.5) \
            and e_m < 1e-10 and e_v < 1e-10
        check(label, ok,
              f"logpdf {e_lp:.1e}, cdf {e_cdf:.1e}, quantile {e_q:.1e}, "
              f"mean {e_m:.1e}, var {e_v:.1e}")

    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nAll comparisons passed.")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
