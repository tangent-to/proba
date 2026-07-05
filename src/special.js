/**
 * Special functions underpinning the distribution numerics:
 * log-gamma, digamma, regularized incomplete gamma/beta and their
 * inverses, erf/erfc, and the inverse normal CDF.
 *
 * Implemented from the standard published algorithms (Lanczos
 * approximation; series and modified-Lentz continued fractions for the
 * incomplete functions; Acklam's rational approximation with Halley
 * refinement for the normal quantile). Accuracy targets are 1e-12 or
 * better across the usual parameter ranges; the scipy comparison suite
 * enforces this.
 */

// Lanczos approximation, g = 7, n = 9 (Godfrey coefficients).
const LANCZOS_G = 7;
const LANCZOS = [
  0.99999999999980993,
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.61502916214059,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7,
];

const LN_SQRT_2PI = 0.9189385332046727; // ln(sqrt(2*pi))
const SQRT2 = Math.SQRT2;

/**
 * Natural log of the absolute value of the gamma function.
 * Exact poles (0, -1, -2, ...) return Infinity.
 *
 * @param {number} x
 * @returns {number} ln|Γ(x)|
 */
export function lgamma(x) {
  if (Number.isNaN(x)) return NaN;
  if (x <= 0 && Number.isInteger(x)) return Infinity;
  if (x < 0.5) {
    // Reflection: Γ(x)Γ(1-x) = π / sin(πx)
    return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  }
  const z = x - 1;
  let sum = LANCZOS[0];
  for (let i = 1; i < LANCZOS.length; i++) {
    sum += LANCZOS[i] / (z + i);
  }
  const t = z + LANCZOS_G + 0.5;
  return LN_SQRT_2PI + (z + 0.5) * Math.log(t) - t + Math.log(sum);
}

/**
 * Digamma function ψ(x) = d/dx ln Γ(x).
 * Recurrence to push the argument above 6, then the asymptotic series.
 *
 * @param {number} x
 * @returns {number}
 */
export function digamma(x) {
  if (Number.isNaN(x)) return NaN;
  if (x <= 0 && Number.isInteger(x)) return NaN; // poles
  let result = 0;
  if (x < 0) {
    // Reflection: ψ(1-x) - ψ(x) = π cot(πx)
    return digamma(1 - x) - Math.PI / Math.tan(Math.PI * x);
  }
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  const inv = 1 / x;
  const inv2 = inv * inv;
  // ψ(x) ~ ln x - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + 1/(240x^8) - 1/(132x^10)
  result += Math.log(x) - 0.5 * inv -
    inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 * (1 / 240 - inv2 / 132))));
  return result;
}

const EPS = Number.EPSILON;
const FPMIN = Number.MIN_VALUE / EPS;
const MAX_ITER = 300;

/**
 * Regularized lower incomplete gamma P(a, x) via its power series.
 * Valid (fast-converging) for x < a + 1.
 */
function gammaPSeries(a, x) {
  let ap = a;
  let sum = 1 / a;
  let del = sum;
  for (let i = 0; i < MAX_ITER * 3; i++) {
    ap += 1;
    del *= x / ap;
    sum += del;
    if (Math.abs(del) < Math.abs(sum) * EPS) break;
  }
  return sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
}

/**
 * Regularized upper incomplete gamma Q(a, x) via modified-Lentz
 * continued fraction. Valid (fast-converging) for x >= a + 1.
 */
function gammaQContinuedFraction(a, x) {
  let b = x + 1 - a;
  let c = 1 / FPMIN;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i <= MAX_ITER * 3; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = b + an / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h * Math.exp(-x + a * Math.log(x) - lgamma(a));
}

/**
 * Regularized lower incomplete gamma P(a, x) = γ(a, x) / Γ(a).
 *
 * @param {number} a - Shape, a > 0
 * @param {number} x - x >= 0
 * @returns {number} P(a, x) in [0, 1]
 */
export function gammainc(a, x) {
  if (Number.isNaN(a) || Number.isNaN(x)) return NaN;
  if (a <= 0) return NaN;
  if (x <= 0) return 0;
  if (x === Infinity) return 1;
  return x < a + 1 ? gammaPSeries(a, x) : 1 - gammaQContinuedFraction(a, x);
}

/**
 * Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x), computed
 * without the cancellation of literally doing 1 - P.
 *
 * @param {number} a - Shape parameter, a > 0
 * @param {number} x - Argument, x >= 0
 * @returns {number} Q(a, x) in [0, 1]
 */
export function gammaincc(a, x) {
  if (Number.isNaN(a) || Number.isNaN(x)) return NaN;
  if (a <= 0) return NaN;
  if (x <= 0) return 1;
  if (x === Infinity) return 0;
  return x < a + 1 ? 1 - gammaPSeries(a, x) : gammaQContinuedFraction(a, x);
}

/**
 * Inverse of the regularized lower incomplete gamma: x with P(a, x) = p.
 * Halley-refined Newton from the Wilson-Hilferty starting point
 * (following the approach in Numerical Recipes §6.2.1, reimplemented).
 *
 * @param {number} p - Probability in [0, 1]
 * @param {number} a - Shape, a > 0
 * @returns {number} x >= 0
 */
export function gammaincInv(p, a) {
  if (Number.isNaN(p) || Number.isNaN(a) || a <= 0) return NaN;
  if (p <= 0) return 0;
  if (p >= 1) return Infinity;

  const a1 = a - 1;
  const gln = lgamma(a);
  let x;

  // Starting guess
  if (a > 1) {
    // Wilson-Hilferty
    const pp = p < 0.5 ? p : 1 - p;
    const t = Math.sqrt(-2 * Math.log(pp));
    let z = (2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481)) - t;
    if (p < 0.5) z = -z;
    x = Math.max(1e-3, a * (1 - 1 / (9 * a) - z / (3 * Math.sqrt(a))) ** 3);
  } else {
    const t = 1 - a * (0.253 + a * 0.12);
    x = p < t ? Math.pow(p / t, 1 / a) : 1 - Math.log(1 - (p - t) / (1 - t));
  }

  // Halley iteration on P(a, x) - p
  let lna1 = 0;
  let afac = 0;
  if (a > 1) {
    lna1 = Math.log(a1);
    afac = Math.exp(a1 * (lna1 - 1) - gln);
  }
  for (let j = 0; j < 20; j++) {
    if (x <= 0) return 0;
    const err = gammainc(a, x) - p;
    let t;
    if (a > 1) {
      t = afac * Math.exp(-(x - a1) + a1 * (Math.log(x) - lna1));
    } else {
      t = Math.exp(-x + a1 * Math.log(x) - gln);
    }
    const u = err / t;
    x -= (t = u / (1 - 0.5 * Math.min(1, u * (a1 / x - 1))));
    if (x <= 0) x = 0.5 * (x + t); // halve rather than going negative
    if (Math.abs(t) < EPS * x) break;
  }
  return x;
}

/**
 * Continued fraction for the regularized incomplete beta (modified Lentz).
 */
function betacf(a, b, x) {
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAX_ITER; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

/**
 * Regularized incomplete beta I_x(a, b).
 *
 * @param {number} a - a > 0
 * @param {number} b - b > 0
 * @param {number} x - x in [0, 1]
 * @returns {number} I_x(a, b) in [0, 1]
 */
export function betainc(a, b, x) {
  if (Number.isNaN(a) || Number.isNaN(b) || Number.isNaN(x)) return NaN;
  if (a <= 0 || b <= 0) return NaN;
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lbeta);
  // Use the continued fraction directly on the side where it converges fast
  if (x < (a + 1) / (a + b + 2)) {
    return front * betacf(a, b, x) / a;
  }
  return 1 - front * betacf(b, a, 1 - x) / b;
}

/**
 * Inverse of the regularized incomplete beta: x with I_x(a, b) = p.
 * Newton iteration with a bisection safeguard.
 *
 * @param {number} p - Probability in [0, 1]
 * @param {number} a - a > 0
 * @param {number} b - b > 0
 * @returns {number} x in [0, 1]
 */
export function betaincInv(p, a, b) {
  if (Number.isNaN(p) || Number.isNaN(a) || Number.isNaN(b)) return NaN;
  if (a <= 0 || b <= 0) return NaN;
  if (p <= 0) return 0;
  if (p >= 1) return 1;

  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);

  // Starting guess (Abramowitz & Stegun 26.5.22)
  let x;
  const yp = normalQuantile(p);
  if (a >= 1 && b >= 1) {
    const lambda = (yp * yp - 3) / 6;
    const h = 2 / (1 / (2 * a - 1) + 1 / (2 * b - 1));
    const w = yp * Math.sqrt(h + lambda) / h -
      (1 / (2 * b - 1) - 1 / (2 * a - 1)) * (lambda + 5 / 6 - 2 / (3 * h));
    x = a / (a + b * Math.exp(2 * w));
  } else {
    const lna = Math.log(a / (a + b));
    const lnb = Math.log(b / (a + b));
    const t = Math.exp(a * lna) / a;
    const u = Math.exp(b * lnb) / b;
    const w = t + u;
    x = p < t / w ? Math.pow(a * w * p, 1 / a) : 1 - Math.pow(b * w * (1 - p), 1 / b);
  }

  // Newton with bisection bracket
  let lo = 0;
  let hi = 1;
  for (let j = 0; j < 50; j++) {
    if (x <= 0 || x >= 1) x = 0.5 * (lo + hi);
    const err = betainc(a, b, x) - p;
    if (err > 0) hi = x;
    else lo = x;
    const pdf = Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lbeta);
    let step = err / pdf;
    let xNew = x - step;
    if (xNew <= lo || xNew >= hi) {
      xNew = 0.5 * (lo + hi);
      step = x - xNew;
    }
    x = xNew;
    if (Math.abs(step) < EPS * x) break;
  }
  return x;
}

/**
 * Error function erf(x), via the regularized incomplete gamma
 * (erf(x) = sign(x) * P(1/2, x^2)), accurate to ~1e-14.
 *
 * @param {number} x
 * @returns {number}
 */
export function erf(x) {
  if (Number.isNaN(x)) return NaN;
  if (x === 0) return 0;
  const v = gammainc(0.5, x * x);
  return x > 0 ? v : -v;
}

/**
 * Complementary error function erfc(x) = 1 - erf(x), computed without
 * cancellation for large positive x.
 *
 * @param {number} x
 * @returns {number} 1 - erf(x)
 */
export function erfc(x) {
  if (Number.isNaN(x)) return NaN;
  if (x >= 0) return gammaincc(0.5, x * x);
  return 1 + gammainc(0.5, x * x);
}

/**
 * Standard normal cumulative distribution function.
 *
 * @param {number} x
 * @returns {number} Φ(x), the probability that a standard normal is <= x
 */
export function normalCdf(x) {
  return 0.5 * erfc(-x / SQRT2);
}

// Acklam's rational approximation coefficients for the inverse normal CDF.
const ACK_A = [
  -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
  1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
];
const ACK_B = [
  -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
  6.680131188771972e+01, -1.328068155288572e+01,
];
const ACK_C = [
  -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
  -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
];
const ACK_D = [
  7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
  3.754408661907416e+00,
];

/**
 * Inverse standard normal CDF (probit), Acklam's approximation refined
 * with one Halley step against erfc — accurate to full double precision.
 *
 * @param {number} p - Probability in (0, 1)
 * @returns {number} z with Φ(z) = p
 */
export function normalQuantile(p) {
  if (Number.isNaN(p)) return NaN;
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;

  const pLow = 0.02425;
  let x;
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((ACK_C[0] * q + ACK_C[1]) * q + ACK_C[2]) * q + ACK_C[3]) * q + ACK_C[4]) * q +
      ACK_C[5]) /
      ((((ACK_D[0] * q + ACK_D[1]) * q + ACK_D[2]) * q + ACK_D[3]) * q + 1);
  } else if (p <= 1 - pLow) {
    const q = p - 0.5;
    const r = q * q;
    x = (((((ACK_A[0] * r + ACK_A[1]) * r + ACK_A[2]) * r + ACK_A[3]) * r + ACK_A[4]) * r +
      ACK_A[5]) * q /
      (((((ACK_B[0] * r + ACK_B[1]) * r + ACK_B[2]) * r + ACK_B[3]) * r + ACK_B[4]) * r + 1);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((ACK_C[0] * q + ACK_C[1]) * q + ACK_C[2]) * q + ACK_C[3]) * q + ACK_C[4]) * q +
      ACK_C[5]) /
      ((((ACK_D[0] * q + ACK_D[1]) * q + ACK_D[2]) * q + ACK_D[3]) * q + 1);
  }

  // One Halley refinement using the exact CDF
  const e = normalCdf(x) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
  x = x - u / (1 + x * u / 2);
  return x;
}

/**
 * Log of the beta function, ln B(a, b) = ln Γ(a) + ln Γ(b) - ln Γ(a + b).
 *
 * @param {number} a - a > 0
 * @param {number} b - b > 0
 * @returns {number} ln B(a, b)
 */
export function lbeta(a, b) {
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

/**
 * Log of the binomial coefficient, ln (n choose k), for non-negative integers.
 *
 * @param {number} n - Total count, n >= 0
 * @param {number} k - Chosen count; returns -Infinity outside 0 <= k <= n
 * @returns {number} ln(n choose k)
 */
export function lchoose(n, k) {
  if (k < 0 || k > n) return -Infinity;
  return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}
