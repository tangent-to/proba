/**
 * Exponential distribution with rate `lambda` (mean 1/lambda).
 */

/**
 * Log density: ln(lambda) - lambda*x for x >= 0, -Infinity for x < 0.
 * The boundary x = 0 is in the support: logpdf(0) = ln(lambda).
 *
 * @param {number} x
 * @param {Object} params - {lambda}
 * @returns {number}
 */
function logpdf(x, { lambda }) {
  if (x < 0) return -Infinity;
  return Math.log(lambda) - lambda * x;
}

/**
 * Exponential distribution, parameterized {lambda} with rate lambda > 0.
 * Support is [0, infinity).
 */
export const exponential = Object.freeze({
  name: 'exponential',
  kind: 'continuous',
  params: ['lambda'],

  support() {
    return [0, Infinity];
  },

  validate({ lambda }) {
    if (!Number.isFinite(lambda) || lambda <= 0) {
      throw new Error(`exponential: lambda must be a positive finite number (got ${lambda})`);
    }
  },

  logpdf,

  pdf(x, params) {
    return Math.exp(logpdf(x, params));
  },

  cdf(x, { lambda }) {
    if (x <= 0) return 0;
    return -Math.expm1(-lambda * x); // 1 - e^{-lambda x} without cancellation
  },

  quantile(p, { lambda }) {
    if (p <= 0) return 0;
    if (p >= 1) return Infinity;
    return -Math.log1p(-p) / lambda;
  },

  /**
   * Gradients of logpdf: dx = -lambda, dlambda = 1/lambda - x.
   */
  dlogpdf(x, { lambda }) {
    if (x < 0) return { dx: NaN, dlambda: NaN };
    return { dx: -lambda, dlambda: 1 / lambda - x };
  },

  sample({ lambda }, rng) {
    return -Math.log1p(-rng.float()) / lambda; // inverse cdf
  },

  sampleN(params, rng, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.sample(params, rng);
    return out;
  },

  mean({ lambda }) {
    return 1 / lambda;
  },

  variance({ lambda }) {
    return 1 / (lambda * lambda);
  },
});
