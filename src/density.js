/**
 * Log-densities as differentiable expressions.
 *
 * `logpdf` is the fast scalar path: plain numbers in, a number out, with the
 * support handled by branches. `logDensity` is the same formula written in
 * `@tangent.to/grad` ops, so that `x` and every parameter may be a `Var`
 * built from a model's free variables, and the result is differentiable in
 * all of them. It is what a sampler's likelihood, a network's loss and a
 * hierarchical prior all reduce to, so it lives here, once, rather than in
 * each package that needs it.
 *
 * Two things differ from `logpdf`, and both are the price of a tape:
 *
 *   - The result is ELEMENTWISE, shaped like `x` (or like the broadcast of
 *     `x` and the parameters: a vector `x` against scalar parameters, or
 *     against vectors of the same length, or a matrix against a vector
 *     spread over its rows). Callers reduce with `sum` or `mean`, or weight
 *     the elements first.
 *   - The SUPPORT is not checked. A branch on the value of a `Var` is not
 *     differentiable, and a sampler keeps its values inside the support by
 *     construction (mc samples bounded parameters through an unconstrained
 *     transform), while an observed value is data the caller validated. Off
 *     the support the formula returns whatever `log` of a negative gives,
 *     not `-Infinity`; the one exception is `uniform` with numeric bounds,
 *     which can check and does.
 *
 * Inside the support, on plain numbers, every function here agrees with its
 * `logpdf` and its gradients with `dlogpdf`; that is the test.
 */

import { add, div, lgamma, log, mul, neg, square, sub, toNested, Var } from '@tangent.to/grad';
// A scalar log-gamma for the constants of the discrete densities: the tape's
// `lgamma` would do, but a constant should not cost a node.
import { lchoose, lgamma as lgammaNumber } from './special.js';

const LN_SQRT_2PI = 0.9189385332046727; // ln sqrt(2 pi)
const LN_SQRT_2_OVER_PI = -0.22579135264472744; // ln sqrt(2 / pi)
const LN2 = Math.LN2;

/** `x` as plain numbers, for a combinatorial constant that has no derivative. @private */
function plain(x) {
  return x instanceof Var ? toNested(x.value) : x;
}

/** Apply a scalar function to a number or (nested) array of numbers. @private */
function mapPlain(x, f) {
  return Array.isArray(x) ? x.map((v) => mapPlain(v, f)) : f(x);
}

/** lgamma(x + 1) of plain integers, as a constant. @private */
function lfactorial(x) {
  return mapPlain(plain(x), (v) => lgammaNumber(v + 1));
}

/** ln B(a, b) as an expression, for parameters that may be Vars. @private */
const lbetaExpr = (a, b) => sub(add(lgamma(a), lgamma(b)), lgamma(add(a, b)));

/** Normal: -log σ - ln√(2π) - z²/2. */
export function normal(x, { mu, sigma }) {
  const z = div(sub(x, mu), sigma);
  return sub(sub(mul(-0.5, square(z)), log(sigma)), LN_SQRT_2PI);
}

/**
 * Uniform: -log(high - low) on [low, high]. With numeric bounds and numeric
 * `x`, an element off the support is -Infinity, as `logpdf` gives.
 */
export function uniform(x, { low, high }) {
  const width = neg(log(sub(high, low)));
  // Shaped like x, carrying no gradient in x: 0·x + width.
  const shaped = add(mul(x, 0), width);
  if (typeof low === 'number' && typeof high === 'number' && !(x instanceof Var)) {
    const off = mapPlain(x, (v) => (v < low || v > high ? -Infinity : 0));
    return add(shaped, off);
  }
  return shaped;
}

/** Exponential: log λ - λx on [0, ∞). */
export function exponential(x, { lambda }) {
  return sub(log(lambda), mul(lambda, x));
}

/** Log-normal: -log x - log σ - ln√(2π) - z²/2 with z = (log x - μ)/σ. */
export function lognormal(x, { mu, sigma }) {
  const lx = log(x);
  const z = div(sub(lx, mu), sigma);
  return sub(sub(sub(mul(-0.5, square(z)), lx), log(sigma)), LN_SQRT_2PI);
}

/** Half-normal: ln√(2/π) - log σ - z²/2 on [0, ∞). */
export function halfnormal(x, { sigma }) {
  const z = div(x, sigma);
  return add(sub(mul(-0.5, square(z)), log(sigma)), LN_SQRT_2_OVER_PI);
}

/** Gamma (shape α, rate β): α log β - lgamma(α) + (α - 1) log x - βx. */
export function gamma(x, { alpha, beta }) {
  const lx = log(x);
  return sub(
    add(mul(alpha, log(beta)), mul(sub(alpha, 1), lx)),
    add(lgamma(alpha), mul(beta, x)),
  );
}

/** Beta: (α - 1) log x + (β - 1) log(1 - x) - ln B(α, β). */
export function beta(x, { alpha, beta: b }) {
  return sub(
    add(mul(sub(alpha, 1), log(x)), mul(sub(b, 1), log(sub(1, x)))),
    lbetaExpr(alpha, b),
  );
}

/**
 * Student-t: lgamma((ν+1)/2) - lgamma(ν/2) - ½ log(νπ) - log σ
 *            - ((ν+1)/2) log(1 + z²/ν).
 */
export function studentT(x, { nu, mu, sigma }) {
  const z = div(sub(x, mu), sigma);
  const halfNu = mul(nu, 0.5);
  const halfNu1 = add(halfNu, 0.5);
  return sub(
    sub(sub(lgamma(halfNu1), lgamma(halfNu)), mul(0.5, log(mul(nu, Math.PI)))),
    add(log(sigma), mul(halfNu1, log(add(1, div(square(z), nu))))),
  );
}

/** Chi-squared: (k/2 - 1) log x - x/2 - (k/2) ln 2 - lgamma(k/2). */
export function chi2(x, { k }) {
  const halfK = mul(k, 0.5);
  return sub(
    sub(mul(sub(halfK, 1), log(x)), mul(x, 0.5)),
    add(mul(halfK, LN2), lgamma(halfK)),
  );
}

/**
 * F: ½d₁ log(d₁/d₂) + (d₁/2 - 1) log x - ((d₁+d₂)/2) log(1 + d₁x/d₂)
 *    - ln B(d₁/2, d₂/2).
 */
export function f(x, { d1, d2 }) {
  const h1 = mul(d1, 0.5);
  const h2 = mul(d2, 0.5);
  return sub(
    add(mul(h1, log(div(d1, d2))), mul(sub(h1, 1), log(x))),
    add(mul(add(h1, h2), log(add(1, div(mul(d1, x), d2)))), lbetaExpr(h1, h2)),
  );
}

/** Bernoulli: x log p + (1 - x) log(1 - p), x in {0, 1}. */
export function bernoulli(x, { p }) {
  return add(mul(x, log(p)), mul(sub(1, x), log(sub(1, p))));
}

/**
 * Binomial: ln C(n, x) + x log p + (n - x) log(1 - p). `n` and `x` are
 * counts, plain numbers; the combinatorial term is a constant.
 */
export function binomial(x, { n, p }) {
  const xs = plain(x);
  const ns = plain(n);
  const lc = Array.isArray(xs)
    ? xs.map((xi, i) => lchoose(Array.isArray(ns) ? ns[i] : ns, xi))
    : lchoose(ns, xs);
  return add(lc, mul(x, log(p)), mul(sub(n, x), log(sub(1, p))));
}

/** Poisson: x log λ - λ - lgamma(x + 1); the last term is a constant in x. */
export function poisson(x, { lambda }) {
  return sub(sub(mul(x, log(lambda)), lambda), lfactorial(x));
}
