// ---
// title: Probability distributions, with gradients
// id: proba-distributions
// ---

// %% [markdown]
/*
`@tangent.to/proba` gives you the full surface of a probability distribution
as a plain object: density, cumulative probability, quantiles, moments,
seedable sampling, and (the part most JavaScript libraries lack) the analytic
gradient of the log density. Every distribution is validated against
scipy.stats to machine precision.
*/

// %% [javascript]

import { normal, gamma, beta, createRng } from 'https://esm.sh/@tangent.to/proba';

// A distribution is a frozen object. Parameters are passed per call, so the
// same object serves every parameterization.
normal.pdf(0, { mu: 0, sigma: 1 });

// %% [markdown]
/*
The standard Normal density, `pdf`, evaluated on a grid. The dot marks
`normal.pdf(0)` — the peak at 0.399.
*/

// %% [javascript]

const stdGrid = d3.range(-4, 4.001, 0.05).map((x) => ({ x, pdf: normal.pdf(x, { mu: 0, sigma: 1 }) }));

const plotStdDensity = Plot.plot({
  height: 240,
  x: { label: 'x' },
  y: { label: 'density', grid: true },
  marks: [
    Plot.ruleY([0]),
    Plot.line(stdGrid, { x: 'x', y: 'pdf', stroke: 'steelblue', strokeWidth: 2 }),
    Plot.dot([{ x: 0, pdf: normal.pdf(0, { mu: 0, sigma: 1 }) }], { x: 'x', y: 'pdf', fill: 'steelblue', r: 4 }),
  ],
});
plotStdDensity;

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
Density (left) and its cumulative probability (right) for the same
`N(100, 15)`. The dashed rule sits at the 95th percentile `q95`; the dot on the
cdf lands at exactly 0.95, the round-trip that `quantile` and `cdf` guarantee.
*/

// %% [javascript]

const iqGrid = d3.range(40, 160.5, 0.5);
const iqDensity = iqGrid.map((x) => ({ x, pdf: normal.pdf(x, params) }));
const iqCdf = iqGrid.map((x) => ({ x, cdf: normal.cdf(x, params) }));

const pdfPlot = Plot.plot({
  width: 360,
  height: 240,
  x: { label: 'x' },
  y: { label: 'density', grid: true },
  marks: [
    Plot.ruleY([0]),
    Plot.line(iqDensity, { x: 'x', y: 'pdf', stroke: 'steelblue', strokeWidth: 2 }),
    Plot.ruleX([q95], { stroke: 'crimson', strokeDasharray: '4 3' }),
  ],
});

const cdfPlot = Plot.plot({
  width: 360,
  height: 240,
  x: { label: 'x' },
  y: { label: 'cumulative probability', grid: true, domain: [0, 1] },
  marks: [
    Plot.line(iqCdf, { x: 'x', y: 'cdf', stroke: 'seagreen', strokeWidth: 2 }),
    Plot.ruleX([q95], { stroke: 'crimson', strokeDasharray: '4 3' }),
    Plot.dot([{ x: q95, cdf: backToP }], { x: 'x', y: 'cdf', fill: 'crimson', r: 4 }),
  ],
});

const iqFigure = document.createElement('div');
iqFigure.style.display = 'flex';
iqFigure.style.flexWrap = 'wrap';
iqFigure.style.gap = '16px';
iqFigure.append(pdfPlot, cdfPlot);
iqFigure;

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
The score, `d/dx log p(x)`, for the standard Normal: a straight line through the
origin with slope `-1/sigma^2`. The dot marks `dlogpdf(1.3).dx = -1.3`.
*/

// %% [javascript]

const scoreGrid = d3.range(-3, 3.001, 0.05).map((x) => ({ x, dx: normal.dlogpdf(x, { mu: 0, sigma: 1 }).dx }));

const plotScore = Plot.plot({
  height: 240,
  x: { label: 'x' },
  y: { label: 'd/dx log p(x)', grid: true },
  marks: [
    Plot.ruleY([0]),
    Plot.ruleX([0]),
    Plot.line(scoreGrid, { x: 'x', y: 'dx', stroke: 'darkorange', strokeWidth: 2 }),
    Plot.dot([{ x: 1.3, dx: normal.dlogpdf(1.3, { mu: 0, sigma: 1 }).dx }], { x: 'x', y: 'dx', fill: 'darkorange', r: 4 }),
  ],
});
plotScore;

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
10,000 `Gamma(2, 0.5)` draws as a histogram, with the analytic `pdf` (scaled to
counts) drawn on top — they line up. The dashed rule is the sample mean; it
falls on the theoretical mean of 4.
*/

// %% [javascript]

const gammaParams = { alpha: 2, beta: 0.5 };
const binWidth = 0.5;
const drawObjs = draws.map((value) => ({ value }));
// pdf(x) * N * binWidth converts the density onto the histogram's count scale.
const gammaCurve = d3
  .range(0, 20, 0.1)
  .map((x) => ({ x, count: gamma.pdf(x, gammaParams) * draws.length * binWidth }));

const plotGammaHist = Plot.plot({
  height: 260,
  x: { label: 'value', domain: [0, 20] },
  y: { label: 'count', grid: true },
  marks: [
    Plot.rectY(drawObjs, Plot.binX({ y: 'count' }, { x: 'value', thresholds: d3.range(0, 20.5, binWidth), fill: '#c9d3df' })),
    Plot.ruleY([0]),
    Plot.line(gammaCurve, { x: 'x', y: 'count', stroke: 'crimson', strokeWidth: 2 }),
    Plot.ruleX([sampleMean], { stroke: 'black', strokeDasharray: '4 3' }),
  ],
});
plotGammaHist;

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

// %% [markdown]
/*
The `Beta(2, 5)` density, skewed toward zero. The solid rule marks the mean, the
dashed rule the median — the mean sits to the right of the median, as it must
for a right-skewed density.
*/

// %% [javascript]

const betaParams = { alpha: 2, beta: 5 };
const betaGrid = d3.range(0.001, 1, 0.005).map((x) => ({ x, pdf: beta.pdf(x, betaParams) }));
const betaMean = beta.mean(betaParams);
const betaMedian = beta.quantile(0.5, betaParams);

const plotBeta = Plot.plot({
  height: 240,
  x: { label: 'x', domain: [0, 1] },
  y: { label: 'density', grid: true },
  marks: [
    Plot.ruleY([0]),
    Plot.areaY(betaGrid, { x: 'x', y: 'pdf', fill: '#d6c3e6' }),
    Plot.line(betaGrid, { x: 'x', y: 'pdf', stroke: 'rebeccapurple', strokeWidth: 2 }),
    Plot.ruleX([betaMean], { stroke: 'rebeccapurple' }),
    Plot.ruleX([betaMedian], { stroke: 'rebeccapurple', strokeDasharray: '4 3' }),
  ],
});
plotBeta;
