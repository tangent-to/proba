/**
 * Contract tests for the seven continuous distributions:
 * normal, uniform, exponential, lognormal, halfnormal, gamma, beta.
 */

import { describe, expect, it } from 'vitest';
import { createRng } from '../src/rng.js';
import { normal } from '../src/normal.js';
import { uniform } from '../src/uniform.js';
import { exponential } from '../src/exponential.js';
import { lognormal } from '../src/lognormal.js';
import { halfnormal } from '../src/halfnormal.js';
import { gamma } from '../src/gamma.js';
import { beta } from '../src/beta.js';

/**
 * Per-distribution fixtures:
 * - params: reference parameter set
 * - points: 5 interior points (cdf/quantile round-trip; first 3 used
 *   for the gradient check)
 * - spots: [x, logpdf] pairs, hand-computed from the closed forms
 * - below/above: out-of-support probes (omitted where the support is
 *   unbounded on that side; ±Infinity is used for the cdf limits)
 * - endpoints: expected quantile(0)/quantile(1)
 * - bad: parameter sets that validate() must reject
 */
const cases = [
  {
    dist: normal,
    params: { mu: 0.5, sigma: 2 },
    points: [-2.5, -0.5, 0.5, 1.5, 4],
    spots: [
      [1.3, -1.6920857137646],
      [-0.4, -1.7133357137646],
    ],
    endpoints: [-Infinity, Infinity],
    bad: [{ mu: 0, sigma: 0 }, { mu: 0, sigma: -1 }, { mu: NaN, sigma: 1 }],
  },
  {
    dist: uniform,
    params: { low: -1, high: 3 },
    points: [-0.9, 0, 1, 2, 2.9],
    spots: [
      [0, -1.3862943611199],
      [2.5, -1.3862943611199],
    ],
    below: -1.5,
    above: 3.5,
    endpoints: [-1, 3],
    bad: [{ low: 2, high: 1 }, { low: 1, high: 1 }, { low: NaN, high: 1 }],
  },
  {
    dist: exponential,
    params: { lambda: 1.5 },
    points: [0.05, 0.3, 0.7, 1.5, 3],
    spots: [
      [2, -2.5945348918918],
      [0, 0.4054651081082],
    ],
    below: -0.5,
    endpoints: [0, Infinity],
    bad: [{ lambda: 0 }, { lambda: -2 }, { lambda: NaN }],
  },
  {
    dist: lognormal,
    params: { mu: 0.2, sigma: 0.8 },
    points: [0.3, 0.8, 1.2, 2, 5],
    spots: [
      [1.5, -1.1342412701939],
      [0.4, -0.753023779757],
    ],
    below: -1,
    endpoints: [0, Infinity],
    bad: [{ mu: 0, sigma: 0 }, { mu: Infinity, sigma: 1 }],
  },
  {
    dist: halfnormal,
    params: { sigma: 1.5 },
    points: [0.1, 0.5, 1, 2, 3.5],
    spots: [
      [0.7, -0.7401453496418],
      [2.1, -1.6112564607529],
    ],
    below: -0.3,
    endpoints: [0, Infinity],
    bad: [{ sigma: 0 }, { sigma: -1 }, { sigma: NaN }],
  },
  {
    dist: gamma,
    params: { alpha: 2.5, beta: 1.5 },
    points: [0.3, 1, 1.7, 2.5, 5],
    spots: [
      [2, -1.2312993293626],
      [0.5, -1.0607408710424],
    ],
    below: -2,
    endpoints: [0, Infinity],
    bad: [{ alpha: 0, beta: 1 }, { alpha: 1, beta: 0 }, { alpha: -1, beta: 1 }],
  },
  {
    dist: beta,
    params: { alpha: 2, beta: 3 },
    points: [0.05, 0.25, 0.4, 0.6, 0.9],
    spots: [
      [0.3, 0.5675839575846],
      [0.62, 0.0717027963216],
    ],
    below: -0.1,
    above: 1.1,
    endpoints: [0, 1],
    bad: [{ alpha: 0, beta: 1 }, { alpha: 1, beta: -1 }, { alpha: NaN, beta: 1 }],
  },
];

/** Central finite difference of f at x with step h. */
function centralDiff(f, x, h = 1e-6) {
  return (f(x + h) - f(x - h)) / (2 * h);
}

/** Assert |a - b| <= tol * max(1, |b|). */
function expectRelClose(a, b, tol) {
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b)));
}

for (const c of cases) {
  const { dist, params } = c;

  describe(dist.name, () => {
    it('logpdf matches hand-computed spot values', () => {
      for (const [x, expected] of c.spots) {
        expect(dist.logpdf(x, params)).toBeCloseTo(expected, 12);
      }
    });

    it('pdf === exp(logpdf)', () => {
      for (const x of c.points) {
        expect(dist.pdf(x, params)).toBe(Math.exp(dist.logpdf(x, params)));
      }
    });

    it('quantile(cdf(x)) round-trips at interior points', () => {
      for (const x of c.points) {
        expect(dist.quantile(dist.cdf(x, params), params)).toBeCloseTo(x, 8);
      }
    });

    it('dlogpdf matches finite differences of logpdf', () => {
      const h = 1e-6;
      for (const x of c.points.slice(0, 3)) {
        const grad = dist.dlogpdf(x, params);
        const numDx = centralDiff((t) => dist.logpdf(t, params), x, h);
        expectRelClose(grad.dx, numDx, 1e-5);
        for (const name of dist.params) {
          const numD = centralDiff(
            (t) => dist.logpdf(x, { ...params, [name]: t }),
            params[name],
            h,
          );
          expectRelClose(grad['d' + name], numD, 1e-5);
        }
      }
    });

    it('sample moments match mean/variance', () => {
      const rng = createRng(12345);
      const n = 20000;
      const draws = dist.sampleN(params, rng, n);
      expect(draws).toHaveLength(n);
      const m = dist.mean(params);
      const v = dist.variance(params);
      const sampleMean = draws.reduce((s, x) => s + x, 0) / n;
      const sampleVar = draws.reduce((s, x) => s + (x - sampleMean) ** 2, 0) / (n - 1);
      expect(Math.abs(sampleMean - m)).toBeLessThanOrEqual(4 * Math.sqrt(v / n));
      expect(Math.abs(sampleVar - v)).toBeLessThanOrEqual(0.15 * v);
    });

    it('out-of-support logpdf is -Infinity and cdf saturates to 0/1', () => {
      const [lo, hi] = dist.support(params);
      if (c.below !== undefined) {
        expect(dist.logpdf(c.below, params)).toBe(-Infinity);
        expect(dist.cdf(c.below, params)).toBe(0);
      }
      if (c.above !== undefined) {
        expect(dist.logpdf(c.above, params)).toBe(-Infinity);
        expect(dist.cdf(c.above, params)).toBe(1);
      }
      // cdf limits at the support edges (finite or infinite)
      expect(dist.cdf(lo === -Infinity ? -Infinity : lo - 1, params)).toBe(0);
      expect(dist.cdf(hi === Infinity ? Infinity : hi, params)).toBe(1);
    });

    it('quantile(0) and quantile(1) return the support endpoints', () => {
      expect(dist.quantile(0, params)).toBe(c.endpoints[0]);
      expect(dist.quantile(1, params)).toBe(c.endpoints[1]);
    });

    it('validate() throws on bad parameters', () => {
      expect(() => dist.validate(params)).not.toThrow();
      for (const bad of c.bad) {
        expect(() => dist.validate(bad)).toThrow(dist.name);
      }
    });
  });
}

describe('boundary limits', () => {
  it('gamma logpdf at x = 0 follows the density limit', () => {
    expect(gamma.logpdf(0, { alpha: 0.5, beta: 2 })).toBe(Infinity);
    expect(gamma.logpdf(0, { alpha: 1, beta: 2 })).toBeCloseTo(Math.log(2), 12);
    expect(gamma.logpdf(0, { alpha: 2, beta: 2 })).toBe(-Infinity);
  });

  it('beta logpdf at x = 0 and x = 1 follows the density limit', () => {
    expect(beta.logpdf(0, { alpha: 0.5, beta: 2 })).toBe(Infinity);
    expect(beta.logpdf(0, { alpha: 1, beta: 2 })).toBeCloseTo(Math.log(2), 12);
    expect(beta.logpdf(0, { alpha: 2, beta: 2 })).toBe(-Infinity);
    expect(beta.logpdf(1, { alpha: 2, beta: 0.5 })).toBe(Infinity);
    expect(beta.logpdf(1, { alpha: 3, beta: 1 })).toBeCloseTo(Math.log(3), 12);
    expect(beta.logpdf(1, { alpha: 2, beta: 2 })).toBe(-Infinity);
  });

  it('gamma sampling handles alpha < 1 (Ahrens-Dieter boost)', () => {
    const rng = createRng(777);
    const params = { alpha: 0.6, beta: 2 };
    const draws = gamma.sampleN(params, rng, 20000);
    const m = draws.reduce((s, x) => s + x, 0) / draws.length;
    expect(draws.every((x) => x >= 0)).toBe(true);
    expect(Math.abs(m - gamma.mean(params))).toBeLessThanOrEqual(
      4 * Math.sqrt(gamma.variance(params) / draws.length),
    );
  });
});
