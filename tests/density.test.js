/**
 * logDensity against logpdf and dlogpdf.
 *
 * The scalar path is scipy-validated; the tape path must be the same formula.
 * So for every distribution, at interior points: the value on plain numbers
 * equals logpdf, and the gradient in x and in each parameter, taken through
 * grad, equals the analytic dlogpdf. Then the shapes: a vector of values
 * against scalar parameters, against vectors, and a matrix against a vector
 * over its rows.
 */

import { describe, expect, it } from 'vitest';
import { sum, valueAndGrad, Var } from '@tangent.to/grad';
import { distributions } from '../src/index.js';

const fixtures = {
  normal: { params: { mu: 1, sigma: 2 }, x: [0.3, 1.5, -2] },
  uniform: { params: { low: -1, high: 3 }, x: [0, 2.5, -0.5] },
  exponential: { params: { lambda: 1.5 }, x: [0.2, 2, 0.9] },
  lognormal: { params: { mu: 0.3, sigma: 0.8 }, x: [0.5, 2, 1.1] },
  halfnormal: { params: { sigma: 1.3 }, x: [0.4, 2, 1] },
  gamma: { params: { alpha: 2.5, beta: 1.5 }, x: [0.7, 3, 1.4] },
  beta: { params: { alpha: 2, beta: 3.5 }, x: [0.2, 0.7, 0.45] },
  studentT: { params: { nu: 4, mu: 0.5, sigma: 1.5 }, x: [-1, 2, 0.3] },
  chi2: { params: { k: 3 }, x: [0.5, 4, 1.7] },
  f: { params: { d1: 5, d2: 7 }, x: [0.5, 2, 1.2] },
  bernoulli: { params: { p: 0.3 }, x: [0, 1, 1] },
  binomial: { params: { n: 10, p: 0.4 }, x: [3, 7, 0] },
  poisson: { params: { lambda: 2.5 }, x: [0, 4, 2] },
};

const close = (a, b, digits = 8) => {
  const scale = 1 + Math.abs(b);
  expect(Math.abs(a - b) / scale).toBeLessThan(10 ** -digits);
};

describe('logDensity equals logpdf inside the support', () => {
  for (const [name, { params, x }] of Object.entries(fixtures)) {
    const dist = distributions[name];
    it(`${name}: at each point, and elementwise over the vector`, () => {
      for (const xi of x) {
        const v = dist.logDensity(xi, params);
        expect(v).toBeInstanceOf(Var);
        expect(v.shape).toEqual([]);
        close(v.data[0], dist.logpdf(xi, params));
      }
      const vec = dist.logDensity(x, params);
      expect(vec.shape).toEqual([x.length]);
      x.forEach((xi, i) => close(vec.data[i], dist.logpdf(xi, params)));
    });
  }
});

describe('logDensity gradients equal dlogpdf', () => {
  for (const [name, { params, x }] of Object.entries(fixtures)) {
    const dist = distributions[name];
    // Integer-valued parameters and counts have no derivative to compare.
    const skip = new Set(name === 'binomial' ? ['n'] : []);
    it(`${name}: in x (continuous) and in every parameter`, () => {
      for (const xi of x) {
        const d = dist.dlogpdf(xi, params);
        if ('dx' in d && dist.kind === 'continuous') {
          const g = valueAndGrad((q) => sum(dist.logDensity(q.x, params)))({ x: xi }).gradient.x;
          close(g, d.dx, 7);
        }
        for (const k of Object.keys(params)) {
          if (skip.has(k) || !(`d${k}` in d)) continue;
          const g = valueAndGrad((q) => sum(dist.logDensity(xi, { ...params, [k]: q.v })))({ v: params[k] })
            .gradient.v;
          close(g, d[`d${k}`], 7);
        }
      }
    });
  }
});

describe('logDensity broadcasts', () => {
  const { normal, poisson } = distributions;

  it('a vector of values against vector parameters, elementwise', () => {
    const v = normal.logDensity([0, 1, 2], { mu: [0, 0.5, 1], sigma: [1, 2, 3] });
    expect(v.shape).toEqual([3]);
    [0, 1, 2].forEach((x, i) => close(v.data[i], normal.logpdf(x, { mu: [0, 0.5, 1][i], sigma: [1, 2, 3][i] })));
  });

  it('a matrix of values against a vector of means over its rows', () => {
    const X = [[0, 1, 2], [1, 1, 1]];
    const mu = [0, 0.5, 1];
    const v = normal.logDensity(X, { mu, sigma: 1.5 });
    expect(v.shape).toEqual([2, 3]);
    X.forEach((row, i) => row.forEach((x, j) => close(v.data[i * 3 + j], normal.logpdf(x, { mu: mu[j], sigma: 1.5 }))));
  });

  it('a Var parameter differentiates, with counts as data', () => {
    const counts = [0, 2, 5];
    const g = valueAndGrad((q) => sum(poisson.logDensity(counts, { lambda: q.l })))({ l: 2 }).gradient.l;
    // d/dλ Σ (x log λ - λ) = Σ x / λ - n
    close(g, 7 / 2 - 3);
  });

  it('uniform with numeric bounds reports -Infinity off the support, as logpdf does', () => {
    const v = distributions.uniform.logDensity([0, 5], { low: -1, high: 3 });
    expect(v.data[0]).toBeCloseTo(-Math.log(4), 12);
    expect(v.data[1]).toBe(-Infinity);
  });
});
