/**
 * Continuous uniform distribution on [low, high].
 */

/**
 * Log density: -ln(high - low) on [low, high] (endpoints inclusive),
 * -Infinity outside.
 *
 * @param {number} x
 * @param {Object} params - {low, high}
 * @returns {number}
 */
function logpdf(x, { low, high }) {
  if (x < low || x > high) return -Infinity;
  return -Math.log(high - low);
}

export const uniform = Object.freeze({
  name: 'uniform',
  kind: 'continuous',
  params: ['low', 'high'],

  support({ low, high }) {
    return [low, high];
  },

  validate({ low, high }) {
    if (!Number.isFinite(low)) {
      throw new Error(`uniform: low must be a finite number (got ${low})`);
    }
    if (!Number.isFinite(high)) {
      throw new Error(`uniform: high must be a finite number (got ${high})`);
    }
    if (!(low < high)) {
      throw new Error(`uniform: low must be strictly less than high (got low=${low}, high=${high})`);
    }
  },

  logpdf,

  pdf(x, params) {
    return Math.exp(logpdf(x, params));
  },

  cdf(x, { low, high }) {
    if (x <= low) return 0;
    if (x >= high) return 1;
    return (x - low) / (high - low);
  },

  quantile(p, { low, high }) {
    if (p <= 0) return low;
    if (p >= 1) return high;
    return low + p * (high - low);
  },

  /**
   * Gradients of logpdf inside the support: dx = 0,
   * dlow = 1/(high-low), dhigh = -1/(high-low).
   * NaN outside the support (the density is not differentiable there).
   */
  dlogpdf(x, { low, high }) {
    if (x < low || x > high) return { dx: NaN, dlow: NaN, dhigh: NaN };
    const w = high - low;
    return { dx: 0, dlow: 1 / w, dhigh: -1 / w };
  },

  sample({ low, high }, rng) {
    return low + (high - low) * rng.float();
  },

  sampleN(params, rng, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.sample(params, rng);
    return out;
  },

  mean({ low, high }) {
    return 0.5 * (low + high);
  },

  variance({ low, high }) {
    const w = high - low;
    return w * w / 12;
  },
});
