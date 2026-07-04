import { describe, it, expect } from 'vitest';
import { studentT } from '../src/studentt.js';
import { chi2 } from '../src/chi2.js';
import { f } from '../src/f.js';
import { bernoulli } from '../src/bernoulli.js';
import { binomial } from '../src/binomial.js';
import { poisson } from '../src/poisson.js';
import { createRng } from '../src/rng.js';

function sampleMean(xs) {
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function sampleVariance(xs) {
  const m = sampleMean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) * (x - m);
  return s / (xs.length - 1);
}

/** Central finite difference of fn at t. */
function centralDiff(fn, t) {
  const h = 1e-5 * Math.max(1, Math.abs(t));
  return (fn(t + h) - fn(t - h)) / (2 * h);
}

function expectGradClose(analytic, numeric) {
  expect(Math.abs(analytic - numeric)).toBeLessThanOrEqual(
    1e-5 * Math.max(1, Math.abs(analytic)),
  );
}

// ---------------------------------------------------------------------------
// (1) logpdf spot constants (validated against scipy.stats to 1e-12)
// ---------------------------------------------------------------------------

describe('logpdf spot constants', () => {
  it('studentT', () => {
    expect(studentT.logpdf(1.3, { nu: 5, mu: 0.5, sigma: 2 }))
      .toBeCloseTo(-1.7562627707927825, 11);
  });

  it('chi2', () => {
    expect(chi2.logpdf(3.7, { k: 4 })).toBeCloseTo(-1.9279615414697118, 11);
  });

  it('f', () => {
    expect(f.logpdf(2.5, { d1: 5, d2: 7 })).toBeCloseTo(-2.3119351228660765, 11);
  });

  it('bernoulli', () => {
    expect(bernoulli.logpdf(1, { p: 0.3 })).toBeCloseTo(-1.2039728043259361, 11);
    expect(bernoulli.logpdf(0, { p: 0.3 })).toBeCloseTo(Math.log(0.7), 12);
  });

  it('binomial', () => {
    expect(binomial.logpdf(7, { n: 20, p: 0.35 })).toBeCloseTo(-1.6906415341280008, 11);
  });

  it('poisson', () => {
    expect(poisson.logpdf(4, { lambda: 2.5 })).toBeCloseTo(-2.0128909028513253, 11);
  });
});

// ---------------------------------------------------------------------------
// (2) pdf === exp(logpdf)
// ---------------------------------------------------------------------------

describe('pdf is exactly exp(logpdf)', () => {
  const cases = [
    [studentT, { nu: 5, mu: 0.5, sigma: 2 }, [-3, 0.5, 1.3, 8]],
    [chi2, { k: 4 }, [0.2, 3.7, 15, -1]],
    [f, { d1: 5, d2: 7 }, [0.1, 2.5, 9, -0.5]],
    [bernoulli, { p: 0.3 }, [0, 1, 0.5]],
    [binomial, { n: 20, p: 0.35 }, [0, 7, 20, 3.5]],
    [poisson, { lambda: 2.5 }, [0, 4, 11, 2.2]],
  ];

  for (const [dist, params, xs] of cases) {
    it(dist.name, () => {
      for (const x of xs) {
        expect(dist.pdf(x, params)).toBe(Math.exp(dist.logpdf(x, params)));
      }
    });
  }
});

// ---------------------------------------------------------------------------
// (3) cdf/quantile round-trips
// ---------------------------------------------------------------------------

describe('cdf/quantile round-trips', () => {
  const ps = [0.01, 0.25, 0.5, 0.75, 0.99];

  const continuous = [
    [studentT, { nu: 5, mu: 0.5, sigma: 2 }],
    [studentT, { nu: 1.5, mu: -3, sigma: 0.7 }],
    [chi2, { k: 4 }],
    [chi2, { k: 0.8 }],
    [f, { d1: 5, d2: 7 }],
    [f, { d1: 1.5, d2: 3 }],
  ];

  for (const [dist, params] of continuous) {
    it(`${dist.name} ${JSON.stringify(params)}`, () => {
      for (const p of ps) {
        const x = dist.quantile(p, params);
        expect(Math.abs(dist.cdf(x, params) - p)).toBeLessThanOrEqual(1e-8);
      }
    });
  }

  it('studentT quantile matches cdf inversion to 1e-9', () => {
    const params = { nu: 5, mu: 0.5, sigma: 2 };
    for (const p of ps) {
      const x = studentT.quantile(p, params);
      expect(Math.abs(studentT.cdf(x, params) - p)).toBeLessThanOrEqual(1e-9);
    }
  });

  it('binomial quantile(cdf(k)) === k', () => {
    const params = { n: 50, p: 0.3 };
    for (const k of [0, 5, 15, 24, 50]) {
      expect(binomial.quantile(binomial.cdf(k, params), params)).toBe(k);
    }
  });

  it('poisson quantile(cdf(k)) === k', () => {
    const params = { lambda: 6.5 };
    for (const k of [0, 2, 6, 12, 20]) {
      expect(poisson.quantile(poisson.cdf(k, params), params)).toBe(k);
    }
  });

  it('cdf spot values match scipy', () => {
    expect(studentT.cdf(1.3, { nu: 5, mu: 0.5, sigma: 2 }))
      .toBeCloseTo(0.6471634425834427, 11);
    expect(chi2.cdf(3.7, { k: 4 })).toBeCloseTo(0.5518740760061613, 11);
    expect(f.cdf(2.5, { d1: 5, d2: 7 })).toBeCloseTo(0.8679937763921592, 11);
    expect(binomial.cdf(7, { n: 20, p: 0.35 })).toBeCloseTo(0.6010266046031639, 11);
    expect(poisson.cdf(4, { lambda: 2.5 })).toBeCloseTo(0.8911780189141513, 11);
  });
});

// ---------------------------------------------------------------------------
// (4) dlogpdf vs central finite differences
// ---------------------------------------------------------------------------

describe('dlogpdf matches finite differences', () => {
  it('studentT', () => {
    const params = { nu: 5, mu: 0.5, sigma: 2 };
    for (const x of [-1.2, 0.8, 3.5]) {
      const g = studentT.dlogpdf(x, params);
      expectGradClose(g.dx, centralDiff((t) => studentT.logpdf(t, params), x));
      expectGradClose(g.dnu, centralDiff((t) => studentT.logpdf(x, { ...params, nu: t }), params.nu));
      expectGradClose(g.dmu, centralDiff((t) => studentT.logpdf(x, { ...params, mu: t }), params.mu));
      expectGradClose(
        g.dsigma,
        centralDiff((t) => studentT.logpdf(x, { ...params, sigma: t }), params.sigma),
      );
    }
  });

  it('chi2', () => {
    const params = { k: 4 };
    for (const x of [0.5, 3.7, 9]) {
      const g = chi2.dlogpdf(x, params);
      expectGradClose(g.dx, centralDiff((t) => chi2.logpdf(t, params), x));
      expectGradClose(g.dk, centralDiff((t) => chi2.logpdf(x, { k: t }), params.k));
    }
  });

  it('f', () => {
    const params = { d1: 5, d2: 7 };
    for (const x of [0.4, 1.3, 4]) {
      const g = f.dlogpdf(x, params);
      expectGradClose(g.dx, centralDiff((t) => f.logpdf(t, params), x));
      expectGradClose(g.dd1, centralDiff((t) => f.logpdf(x, { ...params, d1: t }), params.d1));
      expectGradClose(g.dd2, centralDiff((t) => f.logpdf(x, { ...params, d2: t }), params.d2));
    }
  });

  it('bernoulli (parameter only)', () => {
    for (const p of [0.2, 0.5, 0.85]) {
      for (const x of [0, 1]) {
        const g = bernoulli.dlogpdf(x, { p });
        expectGradClose(g.dp, centralDiff((t) => bernoulli.logpdf(x, { p: t }), p));
      }
    }
  });

  it('binomial (parameter only)', () => {
    const params = { n: 20, p: 0.35 };
    for (const k of [3, 7, 15]) {
      const g = binomial.dlogpdf(k, params);
      expectGradClose(g.dp, centralDiff((t) => binomial.logpdf(k, { n: 20, p: t }), params.p));
    }
  });

  it('poisson (parameter only)', () => {
    const params = { lambda: 2.5 };
    for (const k of [0, 3, 8]) {
      const g = poisson.dlogpdf(k, params);
      expectGradClose(
        g.dlambda,
        centralDiff((t) => poisson.logpdf(k, { lambda: t }), params.lambda),
      );
    }
  });
});

// ---------------------------------------------------------------------------
// (5) seeded sample moments
// ---------------------------------------------------------------------------

describe('seeded sample moments (n = 20000, seed 2468)', () => {
  const N = 20000;

  const cases = [
    [studentT, { nu: 10, mu: 1, sigma: 2 }],
    [chi2, { k: 5 }],
    [f, { d1: 8, d2: 16 }],
    [bernoulli, { p: 0.3 }],
    [binomial, { n: 20, p: 0.35 }],
    [binomial, { n: 150, p: 0.4 }], // quantile-inversion path (n > 64)
    [poisson, { lambda: 4 }],
    [poisson, { lambda: 50 }], // quantile-inversion path (lambda > 30)
  ];

  for (const [dist, params] of cases) {
    it(`${dist.name} ${JSON.stringify(params)}`, () => {
      const rng = createRng(2468);
      const xs = dist.sampleN(params, rng, N);
      expect(xs).toHaveLength(N);
      const mu = dist.mean(params);
      const v = dist.variance(params);
      expect(Math.abs(sampleMean(xs) - mu)).toBeLessThanOrEqual(4 * Math.sqrt(v / N));
      expect(Math.abs(sampleVariance(xs) - v)).toBeLessThanOrEqual(0.15 * v);
    });
  }

  it('discrete samplers return integers in support', () => {
    const rng = createRng(2468);
    for (const x of binomial.sampleN({ n: 150, p: 0.4 }, rng, 500)) {
      expect(Number.isInteger(x)).toBe(true);
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(150);
    }
    for (const x of poisson.sampleN({ lambda: 50 }, rng, 500)) {
      expect(Number.isInteger(x)).toBe(true);
      expect(x).toBeGreaterThanOrEqual(0);
    }
    for (const x of bernoulli.sampleN({ p: 0.3 }, rng, 100)) {
      expect(x === 0 || x === 1).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// (6) support and edge behavior
// ---------------------------------------------------------------------------

describe('support and edge behavior', () => {
  it('support endpoints', () => {
    expect(studentT.support({ nu: 5, mu: 0, sigma: 1 })).toEqual([-Infinity, Infinity]);
    expect(chi2.support({ k: 4 })).toEqual([0, Infinity]);
    expect(f.support({ d1: 5, d2: 7 })).toEqual([0, Infinity]);
    expect(bernoulli.support({ p: 0.3 })).toEqual([0, 1]);
    expect(binomial.support({ n: 20, p: 0.35 })).toEqual([0, 20]);
    expect(poisson.support({ lambda: 2.5 })).toEqual([0, Infinity]);
  });

  it('poisson cdf at non-integer x floors', () => {
    const params = { lambda: 2.5 };
    expect(poisson.cdf(3.7, params)).toBe(poisson.cdf(3, params));
    expect(poisson.cdf(-0.5, params)).toBe(0);
    expect(poisson.logpdf(3.5, params)).toBe(-Infinity);
  });

  it('binomial at k = 0 and k = n, non-integer, out of range', () => {
    const params = { n: 20, p: 0.35 };
    expect(binomial.logpdf(0, params)).toBeCloseTo(20 * Math.log1p(-0.35), 12);
    expect(binomial.logpdf(20, params)).toBeCloseTo(20 * Math.log(0.35), 12);
    expect(binomial.logpdf(7.5, params)).toBe(-Infinity);
    expect(binomial.logpdf(-1, params)).toBe(-Infinity);
    expect(binomial.logpdf(21, params)).toBe(-Infinity);
    expect(binomial.cdf(20, params)).toBe(1);
    expect(binomial.cdf(-1, params)).toBe(0);
    expect(binomial.cdf(6.9, params)).toBe(binomial.cdf(6, params));
  });

  it('binomial p = 0 and p = 1 logpmf edges (no NaN)', () => {
    expect(binomial.logpdf(0, { n: 10, p: 0 })).toBe(0);
    expect(binomial.logpdf(3, { n: 10, p: 0 })).toBe(-Infinity);
    expect(binomial.logpdf(10, { n: 10, p: 1 })).toBe(0);
    expect(binomial.logpdf(4, { n: 10, p: 1 })).toBe(-Infinity);
  });

  it('bernoulli p = 0 and p = 1 logpmf edges (no NaN)', () => {
    expect(bernoulli.logpdf(0, { p: 0 })).toBe(0);
    expect(bernoulli.logpdf(1, { p: 0 })).toBe(-Infinity);
    expect(bernoulli.logpdf(1, { p: 1 })).toBe(0);
    expect(bernoulli.logpdf(0, { p: 1 })).toBe(-Infinity);
    expect(bernoulli.logpdf(0.5, { p: 0.3 })).toBe(-Infinity);
  });

  it('quantile at p = 0 and p = 1 returns support endpoints', () => {
    expect(studentT.quantile(0, { nu: 5, mu: 0.5, sigma: 2 })).toBe(-Infinity);
    expect(studentT.quantile(1, { nu: 5, mu: 0.5, sigma: 2 })).toBe(Infinity);
    expect(studentT.quantile(0.5, { nu: 5, mu: 0.5, sigma: 2 })).toBe(0.5);
    expect(chi2.quantile(0, { k: 4 })).toBe(0);
    expect(chi2.quantile(1, { k: 4 })).toBe(Infinity);
    expect(f.quantile(0, { d1: 5, d2: 7 })).toBe(0);
    expect(f.quantile(1, { d1: 5, d2: 7 })).toBe(Infinity);
    expect(bernoulli.quantile(0, { p: 0.3 })).toBe(0);
    expect(bernoulli.quantile(1, { p: 0.3 })).toBe(1);
    expect(binomial.quantile(0, { n: 20, p: 0.35 })).toBe(0);
    expect(binomial.quantile(1, { n: 20, p: 0.35 })).toBe(20);
    expect(poisson.quantile(0, { lambda: 2.5 })).toBe(0);
  });

  it('out-of-support logpdf is -Infinity, never NaN', () => {
    expect(chi2.logpdf(-1, { k: 4 })).toBe(-Infinity);
    expect(f.logpdf(-0.5, { d1: 5, d2: 7 })).toBe(-Infinity);
    expect(poisson.logpdf(-2, { lambda: 2.5 })).toBe(-Infinity);
    expect(chi2.logpdf(0, { k: 2 })).toBeCloseTo(Math.log(0.5), 12);
    expect(chi2.logpdf(0, { k: 1 })).toBe(Infinity);
    expect(chi2.logpdf(0, { k: 5 })).toBe(-Infinity);
    expect(f.logpdf(0, { d1: 2, d2: 10 })).toBeCloseTo(0, 12); // pdf(0) = 1 for d1 = 2
  });

  it('mean/variance conventions for heavy tails', () => {
    expect(studentT.mean({ nu: 0.5, mu: 3, sigma: 1 })).toBeNaN();
    expect(studentT.mean({ nu: 1.5, mu: 3, sigma: 1 })).toBe(3);
    expect(studentT.variance({ nu: 1.5, mu: 3, sigma: 1 })).toBe(Infinity);
    expect(studentT.variance({ nu: 0.9, mu: 3, sigma: 1 })).toBeNaN();
    expect(studentT.variance({ nu: 10, mu: 3, sigma: 2 })).toBeCloseTo(5, 12);
    expect(f.mean({ d1: 5, d2: 2 })).toBeNaN();
    expect(f.mean({ d1: 5, d2: 10 })).toBeCloseTo(1.25, 12);
    expect(f.variance({ d1: 5, d2: 3 })).toBe(Infinity);
    expect(f.variance({ d1: 5, d2: 2 })).toBeNaN();
  });
});

// ---------------------------------------------------------------------------
// (7) validate() throws on invalid parameters
// ---------------------------------------------------------------------------

describe('validate', () => {
  it('throws with descriptive errors', () => {
    expect(() => studentT.validate({ nu: 0, mu: 0, sigma: 1 })).toThrow(/nu/);
    expect(() => studentT.validate({ nu: 5, mu: NaN, sigma: 1 })).toThrow(/mu/);
    expect(() => studentT.validate({ nu: 5, mu: 0, sigma: -1 })).toThrow(/sigma/);
    expect(() => chi2.validate({ k: 0 })).toThrow(/k/);
    expect(() => chi2.validate({ k: Infinity })).toThrow(/k/);
    expect(() => f.validate({ d1: -1, d2: 7 })).toThrow(/d1/);
    expect(() => f.validate({ d1: 5, d2: 0 })).toThrow(/d2/);
    expect(() => bernoulli.validate({ p: -0.1 })).toThrow(/p/);
    expect(() => bernoulli.validate({ p: 1.1 })).toThrow(/p/);
    expect(() => binomial.validate({ n: 2.5, p: 0.3 })).toThrow(/n/);
    expect(() => binomial.validate({ n: -1, p: 0.3 })).toThrow(/n/);
    expect(() => binomial.validate({ n: 10, p: 1.5 })).toThrow(/p/);
    expect(() => poisson.validate({ lambda: 0 })).toThrow(/lambda/);
    expect(() => poisson.validate({ lambda: -3 })).toThrow(/lambda/);
  });

  it('accepts valid parameters', () => {
    expect(() => studentT.validate({ nu: 5, mu: 0.5, sigma: 2 })).not.toThrow();
    expect(() => chi2.validate({ k: 4 })).not.toThrow();
    expect(() => f.validate({ d1: 5, d2: 7 })).not.toThrow();
    expect(() => bernoulli.validate({ p: 0 })).not.toThrow();
    expect(() => bernoulli.validate({ p: 1 })).not.toThrow();
    expect(() => binomial.validate({ n: 0, p: 0.5 })).not.toThrow();
    expect(() => poisson.validate({ lambda: 2.5 })).not.toThrow();
  });
});
