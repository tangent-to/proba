/**
 * Chi-squared distribution, parameterized {k} with degrees of freedom k > 0.
 */

import { lgamma, digamma, gammainc, gammaincInv } from './special.js';

const LN2 = Math.LN2;

/**
 * One gamma(shape, rate 1) draw via Marsaglia-Tsang, with the standard
 * boost for shape < 1. Local private copy (shared-nothing by design).
 *
 * @param {number} shape
 * @param {Object} rng
 * @returns {number}
 */
function gammaSample(shape, rng) {
  if (shape < 1) {
    return gammaSample(shape + 1, rng) * Math.pow(rng.float(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do {
      x = rng.normal();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng.float();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

/**
 * Chi-squared distribution, parameterized {k} with degrees of freedom k > 0.
 * Support is (0, infinity).
 */
export const chi2 = Object.freeze({
  name: 'chi2',
  kind: 'continuous',
  params: ['k'],

  support() {
    return [0, Infinity];
  },

  validate({ k }) {
    if (!(Number.isFinite(k) && k > 0)) {
      throw new Error(`chi2: k must be a finite number > 0, got ${k}`);
    }
  },

  logpdf(x, { k }) {
    if (Number.isNaN(x)) return -Infinity;
    if (x < 0 || x === Infinity) return -Infinity;
    if (x === 0) {
      // Density boundary: diverges for k < 2, finite for k = 2, zero above.
      if (k < 2) return Infinity;
      if (k === 2) return -LN2;
      return -Infinity;
    }
    return (k / 2 - 1) * Math.log(x) - x / 2 - (k / 2) * LN2 - lgamma(k / 2);
  },

  pdf(x, params) {
    return Math.exp(this.logpdf(x, params));
  },

  cdf(x, { k }) {
    if (Number.isNaN(x)) return NaN;
    if (x <= 0) return 0;
    return gammainc(k / 2, x / 2);
  },

  quantile(p, { k }) {
    if (Number.isNaN(p)) return NaN;
    if (p <= 0) return 0;
    if (p >= 1) return Infinity;
    return 2 * gammaincInv(p, k / 2);
  },

  dlogpdf(x, { k }) {
    return {
      dx: (k / 2 - 1) / x - 0.5,
      dk: 0.5 * Math.log(x) - 0.5 * LN2 - 0.5 * digamma(k / 2),
    };
  },

  sample({ k }, rng) {
    return 2 * gammaSample(k / 2, rng);
  },

  sampleN(params, rng, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.sample(params, rng);
    return out;
  },

  mean({ k }) {
    return k;
  },

  variance({ k }) {
    return 2 * k;
  },
});
