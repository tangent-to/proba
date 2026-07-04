/**
 * Half-normal distribution: |Z| with Z ~ Normal(0, sigma).
 */

import { normalCdf, normalQuantile } from './special.js';

const LN_SQRT_2_OVER_PI = 0.5 * Math.log(2 / Math.PI); // ln(sqrt(2/pi))

/**
 * Log density: 0.5 ln(2/pi) - ln sigma - x^2/(2 sigma^2) for x >= 0,
 * -Infinity for x < 0. The boundary x = 0 is in the support and carries
 * the mode of the density.
 *
 * @param {number} x
 * @param {Object} params - {sigma}
 * @returns {number}
 */
function logpdf(x, { sigma }) {
  if (x < 0) return -Infinity;
  const z = x / sigma;
  return LN_SQRT_2_OVER_PI - Math.log(sigma) - 0.5 * z * z;
}

export const halfnormal = Object.freeze({
  name: 'halfnormal',
  kind: 'continuous',
  params: ['sigma'],

  support() {
    return [0, Infinity];
  },

  validate({ sigma }) {
    if (!Number.isFinite(sigma) || sigma <= 0) {
      throw new Error(`halfnormal: sigma must be a positive finite number (got ${sigma})`);
    }
  },

  logpdf,

  pdf(x, params) {
    return Math.exp(logpdf(x, params));
  },

  cdf(x, { sigma }) {
    if (x <= 0) return 0;
    return 2 * normalCdf(x / sigma) - 1;
  },

  quantile(p, { sigma }) {
    if (p <= 0) return 0;
    if (p >= 1) return Infinity;
    return sigma * normalQuantile(0.5 * (1 + p));
  },

  /**
   * Gradients of logpdf: dx = -x/sigma^2, dsigma = x^2/sigma^3 - 1/sigma.
   */
  dlogpdf(x, { sigma }) {
    if (x < 0) return { dx: NaN, dsigma: NaN };
    const s2 = sigma * sigma;
    return {
      dx: -x / s2,
      dsigma: x * x / (s2 * sigma) - 1 / sigma,
    };
  },

  sample({ sigma }, rng) {
    return sigma * Math.abs(rng.normal());
  },

  sampleN(params, rng, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.sample(params, rng);
    return out;
  },

  mean({ sigma }) {
    return sigma * Math.sqrt(2 / Math.PI);
  },

  variance({ sigma }) {
    return sigma * sigma * (1 - 2 / Math.PI);
  },
});
