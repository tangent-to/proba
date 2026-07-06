/**
 * Poisson distribution, parameterized {lambda} with rate lambda > 0.
 * Support is {0, 1, 2, ...}.
 */

import { lgamma, gammaincc } from './special.js';

/**
 * Poisson distribution, parameterized {lambda} with rate lambda > 0. Support
 * is the non-negative integers {0, 1, 2, ...}.
 */
export const poisson = Object.freeze({
  name: 'poisson',
  kind: 'discrete',
  params: ['lambda'],

  support() {
    return [0, Infinity];
  },

  validate({ lambda }) {
    if (!(Number.isFinite(lambda) && lambda > 0)) {
      throw new Error(`poisson: lambda must be a finite number > 0, got ${lambda}`);
    }
  },

  logpdf(x, { lambda }) {
    if (!Number.isInteger(x) || x < 0) return -Infinity;
    // 0 * log(0) := 0 guards the lambda -> 0 limit at x = 0.
    const kLogLambda = x > 0 ? x * Math.log(lambda) : 0;
    return kLogLambda - lambda - lgamma(x + 1);
  },

  pdf(x, params) {
    return Math.exp(this.logpdf(x, params));
  },

  cdf(x, { lambda }) {
    if (Number.isNaN(x)) return NaN;
    const k = Math.floor(x);
    if (k < 0) return 0;
    if (k === Infinity) return 1;
    // P(X <= k) = Q(k + 1, lambda)
    return gammaincc(k + 1, lambda);
  },

  quantile(prob, params) {
    if (Number.isNaN(prob)) return NaN;
    if (prob <= 0) return 0;
    if (prob >= 1) return Infinity;
    if (this.cdf(0, params) >= prob) return 0;
    // Bracket doubling, then binary search for the smallest k with cdf(k) >= prob.
    let lo = 0;
    let hi = 1;
    while (this.cdf(hi, params) < prob) {
      lo = hi;
      hi *= 2;
    }
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (this.cdf(mid, params) >= prob) hi = mid;
      else lo = mid;
    }
    return hi;
  },

  dlogpdf(x, { lambda }) {
    return { dlambda: x / lambda - 1 };
  },

  sample(params, rng) {
    const { lambda } = params;
    if (lambda <= 30) {
      // Knuth product method
      const limit = Math.exp(-lambda);
      let k = 0;
      let prod = 1;
      do {
        k++;
        prod *= rng.float();
      } while (prod > limit);
      return k - 1;
    }
    // Inversion on the closed-form cdf; quantile is O(log lambda).
    return this.quantile(rng.float(), params);
  },

  sampleN(params, rng, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.sample(params, rng);
    return out;
  },

  mean({ lambda }) {
    return lambda;
  },

  variance({ lambda }) {
    return lambda;
  },
});
