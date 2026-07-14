/**
 * Gamma distribution with shape `alpha` and RATE `beta`
 * (mean = alpha/beta; Bayesian-textbook convention, not scipy's scale).
 */

import { digamma, gammainc, gammaincInv, lgamma } from './special.js';

/**
 * Log density: alpha ln(beta) - lnGamma(alpha) + (alpha-1) ln(x) - beta x
 * for x > 0; -Infinity for x < 0.
 *
 * Boundary x = 0 uses the limit of the density:
 * - alpha < 1: the density diverges, so logpdf(0) = +Infinity;
 * - alpha = 1: the exponential limit, logpdf(0) = ln(beta);
 * - alpha > 1: the density vanishes, logpdf(0) = -Infinity.
 *
 * @param {number} x
 * @param {Object} params - {alpha, beta}
 * @returns {number}
 */
function logpdf(x, { alpha, beta }) {
  if (x < 0 || x === Infinity) return -Infinity;
  if (x === 0) {
    if (alpha < 1) return Infinity;
    if (alpha === 1) return Math.log(beta);
    return -Infinity;
  }
  return alpha * Math.log(beta) - lgamma(alpha) + (alpha - 1) * Math.log(x) - beta * x;
}

/**
 * One draw from Gamma(shape, rate=1) via the Marsaglia-Tsang squeeze
 * method; shapes below 1 are boosted (sample shape+1 and multiply by
 * U^{1/shape}, the Ahrens-Dieter trick).
 *
 * @param {number} alpha - Shape > 0
 * @param {Object} rng - RNG from createRng()
 * @returns {number}
 */
function sampleGammaUnitRate(alpha, rng) {
  if (alpha < 1) {
    const boosted = sampleGammaUnitRate(alpha + 1, rng);
    return boosted * Math.pow(rng.float(), 1 / alpha);
  }
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let z, v;
    do {
      z = rng.normal();
      v = 1 + c * z;
    } while (v <= 0);
    v = v * v * v;
    const u = rng.float();
    if (u < 1 - 0.0331 * z * z * z * z) return d * v;
    if (Math.log(u) < 0.5 * z * z + d * (1 - v + Math.log(v))) return d * v;
  }
}

/**
 * Gamma distribution, parameterized {alpha, beta} with shape alpha > 0 and
 * rate beta > 0. Support is (0, infinity).
 */
export const gamma = Object.freeze({
  name: 'gamma',
  kind: 'continuous',
  params: ['alpha', 'beta'],

  support() {
    return [0, Infinity];
  },

  validate({ alpha, beta }) {
    if (!Number.isFinite(alpha) || alpha <= 0) {
      throw new Error(`gamma: alpha (shape) must be a positive finite number (got ${alpha})`);
    }
    if (!Number.isFinite(beta) || beta <= 0) {
      throw new Error(`gamma: beta (rate) must be a positive finite number (got ${beta})`);
    }
  },

  logpdf,

  pdf(x, params) {
    return Math.exp(logpdf(x, params));
  },

  cdf(x, { alpha, beta }) {
    if (x <= 0) return 0;
    return gammainc(alpha, beta * x);
  },

  quantile(p, { alpha, beta }) {
    if (p <= 0) return 0;
    if (p >= 1) return Infinity;
    return gammaincInv(p, alpha) / beta;
  },

  /**
   * Gradients of logpdf: dx = (alpha-1)/x - beta,
   * dalpha = ln(beta) - digamma(alpha) + ln(x), dbeta = alpha/beta - x.
   */
  dlogpdf(x, { alpha, beta }) {
    if (x <= 0) return { dx: NaN, dalpha: NaN, dbeta: NaN };
    return {
      dx: (alpha - 1) / x - beta,
      dalpha: Math.log(beta) - digamma(alpha) + Math.log(x),
      dbeta: alpha / beta - x,
    };
  },

  sample({ alpha, beta }, rng) {
    return sampleGammaUnitRate(alpha, rng) / beta;
  },

  sampleN(params, rng, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.sample(params, rng);
    return out;
  },

  mean({ alpha, beta }) {
    return alpha / beta;
  },

  variance({ alpha, beta }) {
    return alpha / (beta * beta);
  },
});
