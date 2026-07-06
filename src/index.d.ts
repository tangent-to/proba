/**
 * Type declarations for @tangent.to/proba.
 *
 * The runtime lives in the sibling `.js` modules (plain ESM + JSDoc); this
 * file is the hand-written type surface referenced from `index.js` via
 * `// @ts-self-types`, so JSR/`deno doc` can describe the public API without
 * inferring types from JavaScript. Keep it in sync with the frozen-object
 * contract in CONTRACT.md.
 */

/**
 * Seedable pseudo-random number generator (xoshiro128** seeded through
 * splitmix32). Deterministic across platforms; not cryptographic.
 */
export interface Rng {
  /** Uniform float in [0, 1) with 53-bit resolution. */
  float(): number;
  /** Next unsigned 32-bit integer. */
  int(): number;
  /** Standard normal draw (mean 0, variance 1) via polar Box-Muller. */
  normal(): number;
  /** The seed this generator was created with. */
  seed: number;
}

/**
 * A probability distribution as a frozen plain object. `P` is the parameter
 * object (e.g. `{mu, sigma}`) and `G` is the gradient object returned by
 * `dlogpdf` (`d` + each parameter name, plus `dx` for continuous kinds).
 */
export interface Distribution<P, G> {
  /** Canonical distribution name (e.g. `'normal'`). */
  name: string;
  /** Whether the distribution is `'continuous'` or `'discrete'`. */
  kind: 'continuous' | 'discrete';
  /** Canonical parameter names, in order. */
  params: string[];
  /** Support interval `[min, max]` (inclusive where finite). */
  support(params?: P): [number, number];
  /** Throw an `Error` if the parameters are invalid; otherwise return. */
  validate(params: P): void;
  /** Log density (log pmf for discrete kinds); `-Infinity` outside support, never `NaN`. */
  logpdf(x: number, params: P): number;
  /** Density `exp(logpdf)`; may underflow to 0. */
  pdf(x: number, params: P): number;
  /** Cumulative distribution function `P(X <= x)`. */
  cdf(x: number, params: P): number;
  /** Inverse cdf for `p` in [0, 1]; discrete kinds return the smallest `k` with `cdf(k) >= p`. */
  quantile(p: number, params: P): number;
  /** Analytic gradients of `logpdf` at `x`; `NaN` components outside support. */
  dlogpdf(x: number, params: P): G;
  /** One draw using an `Rng` from {@link createRng}. */
  sample(params: P, rng: Rng): number;
  /** An array of `n` independent draws. */
  sampleN(params: P, rng: Rng, n: number): number[];
  /** Distribution mean; may be `NaN` or `Infinity` where undefined. */
  mean(params: P): number;
  /** Distribution variance; may be `NaN` or `Infinity` where undefined. */
  variance(params: P): number;
}

/** Normal (Gaussian) distribution, parameterized by mean `mu` and standard deviation `sigma`. */
export declare const normal: Distribution<
  { mu: number; sigma: number },
  { dx: number; dmu: number; dsigma: number }
>;

/** Continuous uniform distribution on `[low, high]`. */
export declare const uniform: Distribution<
  { low: number; high: number },
  { dx: number; dlow: number; dhigh: number }
>;

/** Exponential distribution with rate `lambda` (mean `1/lambda`). */
export declare const exponential: Distribution<
  { lambda: number },
  { dx: number; dlambda: number }
>;

/** Lognormal distribution: `ln(X) ~ Normal(mu, sigma)`, with `mu`/`sigma` on the log scale. */
export declare const lognormal: Distribution<
  { mu: number; sigma: number },
  { dx: number; dmu: number; dsigma: number }
>;

/** Half-normal distribution: `|Z|` with `Z ~ Normal(0, sigma)`. */
export declare const halfnormal: Distribution<
  { sigma: number },
  { dx: number; dsigma: number }
>;

/** Gamma distribution with shape `alpha` and RATE `beta` (mean `alpha/beta`). */
export declare const gamma: Distribution<
  { alpha: number; beta: number },
  { dx: number; dalpha: number; dbeta: number }
>;

/** Beta distribution on `[0, 1]` with shape parameters `alpha` and `beta`. */
export declare const beta: Distribution<
  { alpha: number; beta: number },
  { dx: number; dalpha: number; dbeta: number }
>;

/** Student's t distribution (location-scale) with df `nu`, location `mu`, scale `sigma`. */
export declare const studentT: Distribution<
  { nu: number; mu: number; sigma: number },
  { dx: number; dnu: number; dmu: number; dsigma: number }
>;

/** Chi-squared distribution with degrees of freedom `k`. */
export declare const chi2: Distribution<
  { k: number },
  { dx: number; dk: number }
>;

/** F (Fisher-Snedecor) distribution with numerator/denominator df `d1`, `d2`. */
export declare const f: Distribution<
  { d1: number; d2: number },
  { dx: number; dd1: number; dd2: number }
>;

/** Bernoulli distribution with success probability `p` in `[0, 1]`; support `{0, 1}`. */
export declare const bernoulli: Distribution<
  { p: number },
  { dp: number }
>;

/** Binomial distribution with `n` trials and success probability `p`; support `{0, ..., n}`. */
export declare const binomial: Distribution<
  { n: number; p: number },
  { dp: number }
>;

/** Poisson distribution with rate `lambda`; support `{0, 1, 2, ...}`. */
export declare const poisson: Distribution<
  { lambda: number },
  { dlambda: number }
>;

/**
 * Create a seedable {@link Rng}.
 *
 * @param seed Any finite number; omit for a time-based seed.
 */
export declare function createRng(seed?: number): Rng;

/**
 * Special functions underpinning the distribution numerics: log-gamma,
 * digamma, regularized incomplete gamma/beta and their inverses, erf/erfc,
 * and the inverse normal CDF. Accuracy targets are 1e-12 or better.
 */
export declare namespace special {
  /** Natural log of the absolute value of the gamma function; poles return `Infinity`. */
  function lgamma(x: number): number;
  /** Digamma function `ψ(x) = d/dx ln Γ(x)`. */
  function digamma(x: number): number;
  /** Regularized lower incomplete gamma `P(a, x)` in `[0, 1]`. */
  function gammainc(a: number, x: number): number;
  /** Regularized upper incomplete gamma `Q(a, x) = 1 - P(a, x)` in `[0, 1]`. */
  function gammaincc(a: number, x: number): number;
  /** Inverse of the regularized lower incomplete gamma: `x` with `P(a, x) = p`. */
  function gammaincInv(p: number, a: number): number;
  /** Regularized incomplete beta `I_x(a, b)` in `[0, 1]`. */
  function betainc(a: number, b: number, x: number): number;
  /** Inverse of the regularized incomplete beta: `x` with `I_x(a, b) = p`. */
  function betaincInv(p: number, a: number, b: number): number;
  /** Error function `erf(x)`. */
  function erf(x: number): number;
  /** Complementary error function `erfc(x) = 1 - erf(x)`. */
  function erfc(x: number): number;
  /** Standard normal cumulative distribution function `Φ(x)`. */
  function normalCdf(x: number): number;
  /** Inverse standard normal CDF (probit): `z` with `Φ(z) = p`, for `p` in `(0, 1)`. */
  function normalQuantile(p: number): number;
  /** Log of the beta function `ln B(a, b)`. */
  function lbeta(a: number, b: number): number;
  /** Log of the binomial coefficient `ln (n choose k)`; `-Infinity` outside `0 <= k <= n`. */
  function lchoose(n: number, k: number): number;
}

/**
 * Registry of all distributions keyed by name, for dynamic lookup
 * (e.g. model specifications that name distributions as strings).
 */
export declare const distributions: {
  readonly normal: typeof normal;
  readonly uniform: typeof uniform;
  readonly exponential: typeof exponential;
  readonly lognormal: typeof lognormal;
  readonly halfnormal: typeof halfnormal;
  readonly gamma: typeof gamma;
  readonly beta: typeof beta;
  readonly studentT: typeof studentT;
  readonly chi2: typeof chi2;
  readonly f: typeof f;
  readonly bernoulli: typeof bernoulli;
  readonly binomial: typeof binomial;
  readonly poisson: typeof poisson;
};

/**
 * Default export: every distribution spread at the top level, plus the
 * {@link distributions} registry, {@link createRng}, and the {@link special}
 * function namespace.
 */
declare const _default: typeof distributions & {
  distributions: typeof distributions;
  createRng: typeof createRng;
  special: typeof special;
};
export default _default;
