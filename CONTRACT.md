# The distribution contract

Every distribution in `proba` is a **frozen plain object** (not a class —
consumers duck-type against this shape; see the tangent suite conventions).

```js
export const normal = Object.freeze({
  name: 'normal',
  kind: 'continuous',            // or 'discrete'
  params: ['mu', 'sigma'],       // canonical parameter names, in order

  support(params),               // -> [min, max] (inclusive where finite)
  validate(params),              // throws Error on invalid parameters

  logpdf(x, params),             // log density (log pmf for discrete);
                                 //   -Infinity outside the support, never NaN
  pdf(x, params),                // exp(logpdf) — may underflow to 0, that is fine
  cdf(x, params),                // P(X <= x)
  quantile(p, params),           // inverse cdf; p in [0,1]; discrete: smallest
                                 //   k with cdf(k) >= p

  dlogpdf(x, params),            // analytic gradients of logpdf:
                                 //   continuous -> {dx, dmu, dsigma, ...}
                                 //   discrete   -> {dp, ...} (no dx)
                                 //   evaluated AT x; NaN outside support is
                                 //   acceptable, gradients must be exact inside

  sample(params, rng),           // one draw using rng from createRng()
  sampleN(params, rng, n),       // Array of n draws

  mean(params),                  // may be NaN where undefined (e.g. Cauchy-like)
  variance(params),
});
```

Rules:

- **Parameterizations follow the Bayesian-textbook convention** (what PyMC
  and Stan use), NOT scipy's loc/scale everywhere:
  - normal {mu, sigma} · lognormal {mu, sigma} (log-scale params) ·
    halfnormal {sigma} · uniform {low, high} · exponential {lambda} (rate) ·
    gamma {alpha, beta} (shape, RATE) · beta {alpha, beta} ·
    studentT {nu, mu, sigma} · chi2 {k} · f {d1, d2} ·
    bernoulli {p} · binomial {n, p} · poisson {lambda}
  - The scipy comparison suite documents the mapping for each.
- **logpdf is the source of truth**: pdf = exp(logpdf); never implement pdf
  separately.
- **dlogpdf derivative names**: `d` + parameter name (`dmu`, `dsigma`,
  `dalpha`, ...) plus `dx` for continuous distributions. These are what mc's
  HMC/NUTS consume; they must match `numericalGradient` of logpdf to ~1e-6
  (tested).
- **Numerics**: route all cdf/quantile through `src/special.js`
  (lgamma, gammainc, betainc, erf, normalQuantile, ... and their inverses).
  Do not hand-roll approximations in distribution files.
- **Validation**: `logpdf`/`cdf`/`quantile`/`sample` call `validate(params)`
  is NOT required per-call (hot paths); `validate` exists for consumers to
  call at model-build time. Out-of-support x must still return -Infinity/0/1
  correctly without throwing.
- **Edge cases are part of the contract**: logpdf at support boundaries,
  p = 0 and p = 1 quantiles (return support endpoints, possibly ±Infinity),
  degenerate-adjacent parameters (sigma very small, p near 0/1) must not
  return NaN where a correct finite/infinite value exists.
