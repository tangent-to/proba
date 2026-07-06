/**
 * Beta distribution on [0, 1] with shape parameters `alpha` and `beta`.
 */

import { betainc, betaincInv, digamma, lbeta } from './special.js';
import { gamma } from './gamma.js';

/**
 * Log density: (alpha-1) ln(x) + (beta-1) ln(1-x) - lnB(alpha, beta)
 * for 0 < x < 1; -Infinity outside [0, 1].
 *
 * Boundaries use the limit of the density:
 * - x = 0: +Infinity if alpha < 1, ln(beta) if alpha = 1 (uniform-like
 *   edge), -Infinity if alpha > 1; symmetrically at x = 1 with the
 *   roles of alpha and beta swapped.
 *
 * @param {number} x
 * @param {Object} params - {alpha, beta}
 * @returns {number}
 */
function logpdf(x, { alpha, beta }) {
  if (x < 0 || x > 1) return -Infinity;
  if (x === 0) {
    if (alpha < 1) return Infinity;
    if (alpha === 1) return -lbeta(1, beta); // = ln(beta)
    return -Infinity;
  }
  if (x === 1) {
    if (beta < 1) return Infinity;
    if (beta === 1) return -lbeta(alpha, 1); // = ln(alpha)
    return -Infinity;
  }
  return (alpha - 1) * Math.log(x) + (beta - 1) * Math.log1p(-x) - lbeta(alpha, beta);
}

/**
 * Beta distribution, parameterized {alpha, beta} with shape parameters
 * alpha > 0 and beta > 0. Support is the open interval (0, 1).
 */
export const beta = Object.freeze({
  name: 'beta',
  kind: 'continuous',
  params: ['alpha', 'beta'],

  support() {
    return [0, 1];
  },

  validate({ alpha, beta: b }) {
    if (!Number.isFinite(alpha) || alpha <= 0) {
      throw new Error(`beta: alpha must be a positive finite number (got ${alpha})`);
    }
    if (!Number.isFinite(b) || b <= 0) {
      throw new Error(`beta: beta must be a positive finite number (got ${b})`);
    }
  },

  logpdf,

  pdf(x, params) {
    return Math.exp(logpdf(x, params));
  },

  cdf(x, { alpha, beta: b }) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    return betainc(alpha, b, x);
  },

  quantile(p, { alpha, beta: b }) {
    if (p <= 0) return 0;
    if (p >= 1) return 1;
    return betaincInv(p, alpha, b);
  },

  /**
   * Gradients of logpdf: dx = (alpha-1)/x - (beta-1)/(1-x),
   * dalpha = ln(x) - digamma(alpha) + digamma(alpha+beta),
   * dbeta = ln(1-x) - digamma(beta) + digamma(alpha+beta).
   */
  dlogpdf(x, { alpha, beta: b }) {
    if (x <= 0 || x >= 1) return { dx: NaN, dalpha: NaN, dbeta: NaN };
    const dgSum = digamma(alpha + b);
    return {
      dx: (alpha - 1) / x - (b - 1) / (1 - x),
      dalpha: Math.log(x) - digamma(alpha) + dgSum,
      dbeta: Math.log1p(-x) - digamma(b) + dgSum,
    };
  },

  /**
   * Ratio-of-gammas sampler: X = G1/(G1+G2) with
   * G1 ~ Gamma(alpha, 1), G2 ~ Gamma(beta, 1).
   */
  sample({ alpha, beta: b }, rng) {
    const g1 = gamma.sample({ alpha, beta: 1 }, rng);
    const g2 = gamma.sample({ alpha: b, beta: 1 }, rng);
    return g1 / (g1 + g2);
  },

  sampleN(params, rng, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.sample(params, rng);
    return out;
  },

  mean({ alpha, beta: b }) {
    return alpha / (alpha + b);
  },

  variance({ alpha, beta: b }) {
    const s = alpha + b;
    return alpha * b / (s * s * (s + 1));
  },
});
