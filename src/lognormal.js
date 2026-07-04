/**
 * Lognormal distribution: ln(X) ~ Normal(mu, sigma). The parameters are
 * on the log scale (Bayesian-textbook convention, matching PyMC/Stan).
 */

import { normalCdf, normalQuantile } from './special.js';

const LN_SQRT_2PI = 0.9189385332046727; // ln(sqrt(2*pi))

/**
 * Log density: -ln x - ln sigma - 0.5 ln(2 pi) - z^2/2 with
 * z = (ln x - mu)/sigma, for x > 0; -Infinity for x <= 0 (the density
 * vanishes at 0 in the limit).
 *
 * @param {number} x
 * @param {Object} params - {mu, sigma}
 * @returns {number}
 */
function logpdf(x, { mu, sigma }) {
  if (x <= 0) return -Infinity;
  const lx = Math.log(x);
  const z = (lx - mu) / sigma;
  return -lx - Math.log(sigma) - LN_SQRT_2PI - 0.5 * z * z;
}

export const lognormal = Object.freeze({
  name: 'lognormal',
  kind: 'continuous',
  params: ['mu', 'sigma'],

  support() {
    return [0, Infinity];
  },

  validate({ mu, sigma }) {
    if (!Number.isFinite(mu)) {
      throw new Error(`lognormal: mu must be a finite number (got ${mu})`);
    }
    if (!Number.isFinite(sigma) || sigma <= 0) {
      throw new Error(`lognormal: sigma must be a positive finite number (got ${sigma})`);
    }
  },

  logpdf,

  pdf(x, params) {
    return Math.exp(logpdf(x, params));
  },

  cdf(x, { mu, sigma }) {
    if (x <= 0) return 0;
    return normalCdf((Math.log(x) - mu) / sigma);
  },

  quantile(p, { mu, sigma }) {
    if (p <= 0) return 0;
    if (p >= 1) return Infinity;
    return Math.exp(mu + sigma * normalQuantile(p));
  },

  /**
   * Gradients of logpdf with z = (ln x - mu)/sigma:
   * dx = -1/x - z/(sigma*x), dmu = z/sigma, dsigma = (z^2 - 1)/sigma.
   */
  dlogpdf(x, { mu, sigma }) {
    if (x <= 0) return { dx: NaN, dmu: NaN, dsigma: NaN };
    const z = (Math.log(x) - mu) / sigma;
    return {
      dx: -(1 + z / sigma) / x,
      dmu: z / sigma,
      dsigma: (z * z - 1) / sigma,
    };
  },

  sample({ mu, sigma }, rng) {
    return Math.exp(mu + sigma * rng.normal());
  },

  sampleN(params, rng, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.sample(params, rng);
    return out;
  },

  mean({ mu, sigma }) {
    return Math.exp(mu + 0.5 * sigma * sigma);
  },

  variance({ mu, sigma }) {
    const s2 = sigma * sigma;
    return Math.expm1(s2) * Math.exp(2 * mu + s2);
  },
});
