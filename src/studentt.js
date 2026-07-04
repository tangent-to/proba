/**
 * Student's t distribution (location-scale), parameterized {nu, mu, sigma}
 * with degrees of freedom nu > 0, location mu, and scale sigma > 0.
 */

import { lgamma, digamma, betainc, betaincInv } from './special.js';

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

export const studentT = Object.freeze({
  name: 'studentT',
  kind: 'continuous',
  params: ['nu', 'mu', 'sigma'],

  support() {
    return [-Infinity, Infinity];
  },

  validate({ nu, mu, sigma }) {
    if (!(Number.isFinite(nu) && nu > 0)) {
      throw new Error(`studentT: nu must be a finite number > 0, got ${nu}`);
    }
    if (!Number.isFinite(mu)) {
      throw new Error(`studentT: mu must be a finite number, got ${mu}`);
    }
    if (!(Number.isFinite(sigma) && sigma > 0)) {
      throw new Error(`studentT: sigma must be a finite number > 0, got ${sigma}`);
    }
  },

  logpdf(x, { nu, mu, sigma }) {
    if (!Number.isFinite(x)) return -Infinity;
    const z = (x - mu) / sigma;
    return lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * Math.log(nu * Math.PI) -
      Math.log(sigma) - ((nu + 1) / 2) * Math.log1p(z * z / nu);
  },

  pdf(x, params) {
    return Math.exp(this.logpdf(x, params));
  },

  cdf(x, { nu, mu, sigma }) {
    const t = (x - mu) / sigma;
    if (t === 0) return 0.5;
    if (Number.isNaN(t)) return NaN;
    const w = betainc(nu / 2, 0.5, nu / (nu + t * t));
    return t > 0 ? 1 - 0.5 * w : 0.5 * w;
  },

  quantile(p, { nu, mu, sigma }) {
    if (Number.isNaN(p)) return NaN;
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;
    if (p === 0.5) return mu;
    const tail = p < 0.5 ? 2 * p : 2 * (1 - p);
    const w = betaincInv(tail, nu / 2, 0.5); // w = nu / (nu + t^2)
    const t = Math.sqrt(nu * (1 - w) / w);
    return mu + sigma * (p < 0.5 ? -t : t);
  },

  dlogpdf(x, { nu, mu, sigma }) {
    const z = (x - mu) / sigma;
    const w = 1 + z * z / nu;
    const dx = -(nu + 1) * z / (nu * w * sigma);
    return {
      dx,
      dnu: 0.5 * (digamma((nu + 1) / 2) - digamma(nu / 2) - 1 / nu - Math.log(w) +
        (nu + 1) * z * z / (nu * nu * w)),
      dmu: -dx,
      dsigma: -1 / sigma + (nu + 1) * z * z / (nu * w * sigma),
    };
  },

  sample({ nu, mu, sigma }, rng) {
    const chi2draw = 2 * gammaSample(nu / 2, rng); // chi-square(nu)
    return mu + sigma * (rng.normal() / Math.sqrt(chi2draw / nu));
  },

  sampleN(params, rng, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.sample(params, rng);
    return out;
  },

  mean({ nu, mu }) {
    return nu > 1 ? mu : NaN;
  },

  variance({ nu, sigma }) {
    if (nu > 2) return sigma * sigma * nu / (nu - 2);
    return nu > 1 ? Infinity : NaN;
  },
});
