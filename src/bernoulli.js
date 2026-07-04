/**
 * Bernoulli distribution, parameterized {p} with success probability
 * p in [0, 1]. Support is {0, 1}.
 */

export const bernoulli = Object.freeze({
  name: 'bernoulli',
  kind: 'discrete',
  params: ['p'],

  support() {
    return [0, 1];
  },

  validate({ p }) {
    if (!(Number.isFinite(p) && p >= 0 && p <= 1)) {
      throw new Error(`bernoulli: p must be a number in [0, 1], got ${p}`);
    }
  },

  logpdf(x, { p }) {
    // log pmf; 0 * log(0) := 0 at the p = 0 / p = 1 edges.
    if (x === 1) return Math.log(p);
    if (x === 0) return p > 0 ? Math.log1p(-p) : 0; // avoid -0 at p = 0
    return -Infinity;
  },

  pdf(x, params) {
    return Math.exp(this.logpdf(x, params));
  },

  cdf(x, { p }) {
    if (Number.isNaN(x)) return NaN;
    if (x < 0) return 0;
    if (x < 1) return 1 - p;
    return 1;
  },

  quantile(prob, { p }) {
    if (Number.isNaN(prob)) return NaN;
    if (prob <= 0) return 0;
    if (prob > 1 - p) return 1;
    return 0;
  },

  dlogpdf(x, { p }) {
    return { dp: x / p - (1 - x) / (1 - p) };
  },

  sample({ p }, rng) {
    return rng.float() < p ? 1 : 0;
  },

  sampleN(params, rng, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.sample(params, rng);
    return out;
  },

  mean({ p }) {
    return p;
  },

  variance({ p }) {
    return p * (1 - p);
  },
});
