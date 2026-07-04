# tangent/proba

Probability distributions for JavaScript (ESM). Browser-first, zero
dependencies, runs in Node.js and Deno. The probability foundation of the
[tangent suite](https://github.com/tangent-to) — consumed by
[tangent/ds](https://github.com/tangent-to/ds) and
[tangent/mc](https://github.com/tangent-to/mc), MIT-licensed so anyone else
can build on it too.

13 distributions — normal, uniform, exponential, lognormal, halfnormal,
gamma, beta, Student t, χ², F, Bernoulli, binomial, Poisson — each with:

- **`logpdf`** (the source of truth) and **analytic gradients**
  (`dlogpdf` with respect to the value *and every parameter*) — what
  HMC/NUTS samplers need, no autodiff framework required
- `pdf`, `cdf`, `quantile` built on a validated special-function core
  (log-gamma, digamma, regularized incomplete gamma/beta and inverses,
  erf/erfc, inverse normal CDF)
- seedable sampling (`createRng`, xoshiro128**), moments, `support`,
  `validate`

## Install

```bash
npm install @tangent.to/proba     # npm
deno add jsr:@tangent/proba       # Deno / JSR
```

## Usage

```javascript
import { createRng, gamma, normal } from '@tangent.to/proba';

normal.logpdf(1.3, { mu: 0, sigma: 1 });      // -1.7639385332046727
normal.dlogpdf(1.3, { mu: 0, sigma: 1 });     // { dx: -1.3, dmu: 1.3, dsigma: 0.69 }
gamma.quantile(0.95, { alpha: 2, beta: 0.5 }); // rate parameterization

const rng = createRng(42);                    // seeded, reproducible
gamma.sampleN({ alpha: 2, beta: 0.5 }, rng, 1000);
```

Distributions use the Bayesian-textbook parameterizations (PyMC/Stan
style): `gamma {alpha, beta}` is shape/**rate**, `exponential {lambda}` is
rate, `lognormal {mu, sigma}` are log-scale parameters. See
[CONTRACT.md](./CONTRACT.md) for the full contract every distribution
satisfies, including edge-case behavior.

## Validation against scipy

`tests_compare-to-scipy/` checks every distribution against `scipy.stats`
on grids spanning both tails, two parameter sets each: logpdf/logpmf and
cdf agree to ~1e-13 or better, quantiles to 1e-10 (discrete quantiles match
exactly), moments to machine precision. Analytic gradients are verified
against finite differences in the unit suite. Requires
[uv](https://docs.astral.sh/uv/) and Node:

```bash
npm run test:scipy
```

## Scope

Distribution mathematics only: densities, gradients, cdf/quantile,
sampling, moments. No model objects, no fitting, no inference — those live
in the packages that consume proba (ds for frequentist statistics, mc for
Bayesian inference). New distributions are welcome when a consumer needs
them; speculative additions are not.

## License

MIT.
