/**
 * Normal (Gaussian) distribution, parameterized by mean `mu` and
 * standard deviation `sigma` (Bayesian-textbook convention).
 */

import { normalCdf, normalQuantile } from './special.js';

const LN_SQRT_2PI = 0.9189385332046727; // ln(sqrt(2*pi))

/**
 * Log density of the normal distribution. Finite for every finite x
 * (the support is the whole real line, so this is never -Infinity).
 *
 * @param {number} x
 * @param {Object} params - {mu, sigma}
 * @returns {number}
 */
function logpdf(x, { mu, sigma }) {
  const z = (x - mu) / sigma;
  return -Math.log(sigma) - LN_SQRT_2PI - 0.5 * z * z;
}

export const normal = Object.freeze({
  name: 'normal',
  kind: 'continuous',
  params: ['mu', 'sigma'],

  support() {
    return [-Infinity, Infinity];
  },

  validate({ mu, sigma }) {
    if (!Number.isFinite(mu)) {
      throw new Error(`normal: mu must be a finite number (got ${mu})`);
    }
    if (!Number.isFinite(sigma) || sigma <= 0) {
      throw new Error(`normal: sigma must be a positive finite number (got ${sigma})`);
    }
  },

  logpdf,

  pdf(x, params) {
    return Math.exp(logpdf(x, params));
  },

  cdf(x, { mu, sigma }) {
    return normalCdf((x - mu) / sigma);
  },

  quantile(p, { mu, sigma }) {
    return mu + sigma * normalQuantile(p);
  },

  /**
   * Gradients of logpdf: dx = -(x-mu)/sigma^2, dmu = (x-mu)/sigma^2,
   * dsigma = ((x-mu)^2/sigma^2 - 1)/sigma.
   */
  dlogpdf(x, { mu, sigma }) {
    const s2 = sigma * sigma;
    const d = (x - mu) / s2;
    return {
      dx: -d,
      dmu: d,
      dsigma: ((x - mu) * (x - mu) / s2 - 1) / sigma,
    };
  },

  sample({ mu, sigma }, rng) {
    return mu + sigma * rng.normal();
  },

  sampleN(params, rng, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.sample(params, rng);
    return out;
  },

  mean({ mu }) {
    return mu;
  },

  variance({ sigma }) {
    return sigma * sigma;
  },
});
