// ---
// title: Probability distributions, with gradients
// id: proba-distributions
// ---

// %% [markdown]
/*
# Probability distributions

`@tangent.to/proba` gives you the full surface of a probability distribution
as a plain object: density, cumulative probability, quantiles, moments,
seedable sampling, and (the part most JavaScript libraries lack) the analytic
gradient of the log density. Every distribution is validated against
scipy.stats to machine precision.

This notebook imports the local build. Once the package is published you would
import it from a CDN instead:

    import { normal, gamma, createRng } from 'https://esm.sh/@tangent.to/proba';
*/

// %% [javascript]

import { normal, gamma, beta, createRng } from '../dist/index.js';

// A distribution is a frozen object. Parameters are passed per call, so the
// same object serves every parameterization.
normal.pdf(0, { mu: 0, sigma: 1 });

// %% [markdown]
/*
## Density, cumulative probability, quantiles

The three functions every distribution provides. `quantile` is the exact
inverse of `cdf`, so they round-trip.
*/

// %% [javascript]

const params = { mu: 100, sigma: 15 }; // an IQ-like scale
const q95 = normal.quantile(0.95, params); // the 95th percentile
const backToP = normal.cdf(q95, params); // returns 0.95

({
  density_at_115: normal.pdf(115, params),
  ninetyFifth_percentile: q95,
  cdf_roundtrip: backToP,
});

// %% [markdown]
/*
## Analytic gradients

`dlogpdf` returns the gradient of the log density with respect to the value
and every parameter. This is what gradient-based samplers (Hamiltonian Monte
Carlo, NUTS) and maximum-likelihood fitting need, and computing it by hand for
each distribution is exactly the tedium proba removes. The gradient is exact,
not a finite-difference approximation.
*/

// %% [javascript]

// For a Normal, d/dx log p(x) = -(x - mu) / sigma^2.
normal.dlogpdf(1.3, { mu: 0, sigma: 1 });
// { dx: -1.3, dmu: 1.3, dsigma: 0.69 }

// %% [markdown]
/*
## Seedable sampling

`createRng(seed)` gives a reproducible stream (xoshiro128\*\*). Pass it to
`sample` for one draw or `sampleN` for many. The same seed always yields the
same sequence, so simulations are reproducible across machines.
*/

// %% [javascript]

const rng = createRng(42);
const draws = gamma.sampleN({ alpha: 2, beta: 0.5 }, rng, 10000);

const sampleMean = draws.reduce((a, b) => a + b, 0) / draws.length;

({
  first_three: draws.slice(0, 3),
  sample_mean: sampleMean,
  theoretical_mean: gamma.mean({ alpha: 2, beta: 0.5 }), // alpha / beta = 4
});

// %% [markdown]
/*
## Parameterizations follow the textbook conventions

proba uses the Bayesian convention: `gamma` is shape and RATE (mean =
alpha / beta), `beta` is the two shape parameters. Each distribution documents
its own parameter names, matching the symbol you would find in a statistics
text.
*/

// %% [javascript]

// Beta(2, 5): skewed toward zero, mean = 2 / (2 + 5)
({
  mean: beta.mean({ alpha: 2, beta: 5 }),
  median: beta.quantile(0.5, { alpha: 2, beta: 5 }),
  density_at_0_2: beta.pdf(0.2, { alpha: 2, beta: 5 }),
});
