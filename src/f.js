/**
 * F (Fisher-Snedecor) distribution, parameterized {d1, d2} with numerator
 * and denominator degrees of freedom d1 > 0, d2 > 0.
 */

import { digamma, lbeta, betainc, betaincInv } from './special.js';

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

export const f = Object.freeze({
  name: 'f',
  kind: 'continuous',
  params: ['d1', 'd2'],

  support() {
    return [0, Infinity];
  },

  validate({ d1, d2 }) {
    if (!(Number.isFinite(d1) && d1 > 0)) {
      throw new Error(`f: d1 must be a finite number > 0, got ${d1}`);
    }
    if (!(Number.isFinite(d2) && d2 > 0)) {
      throw new Error(`f: d2 must be a finite number > 0, got ${d2}`);
    }
  },

  logpdf(x, { d1, d2 }) {
    if (Number.isNaN(x)) return -Infinity;
    if (x < 0 || x === Infinity) return -Infinity;
    if (x === 0) {
      // Density boundary: diverges for d1 < 2, finite for d1 = 2, zero above.
      if (d1 < 2) return Infinity;
      if (d1 === 2) return 0.5 * d1 * Math.log(d1 / d2) - lbeta(d1 / 2, d2 / 2);
      return -Infinity;
    }
    return 0.5 * d1 * Math.log(d1 / d2) + (d1 / 2 - 1) * Math.log(x) -
      ((d1 + d2) / 2) * Math.log1p(d1 * x / d2) - lbeta(d1 / 2, d2 / 2);
  },

  pdf(x, params) {
    return Math.exp(this.logpdf(x, params));
  },

  cdf(x, { d1, d2 }) {
    if (Number.isNaN(x)) return NaN;
    if (x <= 0) return 0;
    if (x === Infinity) return 1;
    return betainc(d1 / 2, d2 / 2, d1 * x / (d1 * x + d2));
  },

  quantile(p, { d1, d2 }) {
    if (Number.isNaN(p)) return NaN;
    if (p <= 0) return 0;
    if (p >= 1) return Infinity;
    const y = betaincInv(p, d1 / 2, d2 / 2); // y = d1 x / (d1 x + d2)
    return d2 * y / (d1 * (1 - y));
  },

  dlogpdf(x, { d1, d2 }) {
    const u = 1 + d1 * x / d2;
    return {
      dx: (d1 / 2 - 1) / x - (d1 + d2) * d1 / (2 * d2 * u),
      dd1: 0.5 * (Math.log(d1 / d2) + 1 + Math.log(x) - Math.log(u) -
        (d1 + d2) * x / (d2 * u) - digamma(d1 / 2) + digamma((d1 + d2) / 2)),
      dd2: -d1 / (2 * d2) - 0.5 * Math.log(u) + (d1 + d2) * d1 * x / (2 * d2 * d2 * u) -
        0.5 * digamma(d2 / 2) + 0.5 * digamma((d1 + d2) / 2),
    };
  },

  sample({ d1, d2 }, rng) {
    const num = 2 * gammaSample(d1 / 2, rng) / d1; // chi2(d1) / d1
    const den = 2 * gammaSample(d2 / 2, rng) / d2; // chi2(d2) / d2
    return num / den;
  },

  sampleN(params, rng, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.sample(params, rng);
    return out;
  },

  mean({ d2 }) {
    return d2 > 2 ? d2 / (d2 - 2) : NaN;
  },

  variance({ d1, d2 }) {
    if (d2 > 4) {
      return 2 * d2 * d2 * (d1 + d2 - 2) / (d1 * (d2 - 2) * (d2 - 2) * (d2 - 4));
    }
    return d2 > 2 ? Infinity : NaN;
  },
});
