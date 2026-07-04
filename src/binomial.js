/**
 * Binomial distribution, parameterized {n, p} with number of trials
 * n (non-negative integer) and success probability p in [0, 1].
 * Support is {0, 1, ..., n}.
 */

import { lchoose, betainc } from './special.js';

export const binomial = Object.freeze({
  name: 'binomial',
  kind: 'discrete',
  params: ['n', 'p'],

  support({ n }) {
    return [0, n];
  },

  validate({ n, p }) {
    if (!(Number.isInteger(n) && n >= 0)) {
      throw new Error(`binomial: n must be a non-negative integer, got ${n}`);
    }
    if (!(Number.isFinite(p) && p >= 0 && p <= 1)) {
      throw new Error(`binomial: p must be a number in [0, 1], got ${p}`);
    }
  },

  logpdf(x, { n, p }) {
    if (!Number.isInteger(x) || x < 0 || x > n) return -Infinity;
    // 0 * log(0) := 0 at the p = 0 / p = 1 edges; lchoose is exactly 0 at the
    // support endpoints (the Lanczos round-trip would leave ~1e-16 fuzz there).
    const lc = x === 0 || x === n ? 0 : lchoose(n, x);
    const succ = x > 0 ? x * Math.log(p) : 0;
    const fail = x < n ? (n - x) * Math.log1p(-p) : 0;
    return lc + succ + fail;
  },

  pdf(x, params) {
    return Math.exp(this.logpdf(x, params));
  },

  cdf(x, { n, p }) {
    if (Number.isNaN(x)) return NaN;
    const k = Math.floor(x);
    if (k < 0) return 0;
    if (k >= n) return 1;
    // P(X <= k) = I_{1-p}(n - k, k + 1)
    return betainc(n - k, k + 1, 1 - p);
  },

  quantile(prob, params) {
    const { n } = params;
    if (Number.isNaN(prob)) return NaN;
    if (prob <= 0) return 0;
    if (prob >= 1) return n;
    if (this.cdf(0, params) >= prob) return 0;
    // Bracket doubling, then binary search for the smallest k with cdf(k) >= prob.
    let lo = 0;
    let hi = 1;
    while (hi < n && this.cdf(hi, params) < prob) {
      lo = hi;
      hi = Math.min(2 * hi, n);
    }
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (this.cdf(mid, params) >= prob) hi = mid;
      else lo = mid;
    }
    return hi;
  },

  dlogpdf(x, { n, p }) {
    return { dp: x / p - (n - x) / (1 - p) };
  },

  sample(params, rng) {
    const { n, p } = params;
    if (n <= 64) {
      let k = 0;
      for (let i = 0; i < n; i++) {
        if (rng.float() < p) k++;
      }
      return k;
    }
    // Inversion on the closed-form cdf; quantile is O(log n).
    return this.quantile(rng.float(), params);
  },

  sampleN(params, rng, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.sample(params, rng);
    return out;
  },

  mean({ n, p }) {
    return n * p;
  },

  variance({ n, p }) {
    return n * p * (1 - p);
  },
});
