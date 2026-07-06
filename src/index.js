/**
 * @tangent.to/proba - Probability distributions for JavaScript (ESM)
 *
 * Every distribution follows the frozen-object contract in CONTRACT.md:
 * logpdf (source of truth) with analytic gradients (dlogpdf), pdf, cdf,
 * quantile, seedable sampling, moments, support and validation.
 * MIT-licensed infrastructure of the tangent suite; consumed by
 * @tangent.to/ds and @tangent.to/mc.
 */

export { normal } from './normal.js';
export { uniform } from './uniform.js';
export { exponential } from './exponential.js';
export { lognormal } from './lognormal.js';
export { halfnormal } from './halfnormal.js';
export { gamma } from './gamma.js';
export { beta } from './beta.js';
export { studentT } from './studentt.js';
export { chi2 } from './chi2.js';
export { f } from './f.js';
export { bernoulli } from './bernoulli.js';
export { binomial } from './binomial.js';
export { poisson } from './poisson.js';

export { createRng } from './rng.js';
export * as special from './special.js';

import { normal } from './normal.js';
import { uniform } from './uniform.js';
import { exponential } from './exponential.js';
import { lognormal } from './lognormal.js';
import { halfnormal } from './halfnormal.js';
import { gamma } from './gamma.js';
import { beta } from './beta.js';
import { studentT } from './studentt.js';
import { chi2 } from './chi2.js';
import { f } from './f.js';
import { bernoulli } from './bernoulli.js';
import { binomial } from './binomial.js';
import { poisson } from './poisson.js';
import { createRng } from './rng.js';
import * as special from './special.js';

/**
 * Registry of all distributions keyed by name, for dynamic lookup
 * (e.g. model specifications that name distributions as strings).
 */
export const distributions = Object.freeze({
  normal,
  uniform,
  exponential,
  lognormal,
  halfnormal,
  gamma,
  beta,
  studentT,
  chi2,
  f,
  bernoulli,
  binomial,
  poisson,
});

export default {
  ...distributions,
  distributions,
  createRng,
  special,
};
