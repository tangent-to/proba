import { describe, expect, it } from 'vitest';
import {
  betainc,
  betaincInv,
  digamma,
  erf,
  erfc,
  gammainc,
  gammaincc,
  gammaincInv,
  lbeta,
  lchoose,
  lgamma,
  normalCdf,
  normalQuantile,
} from '../src/special.js';

// Reference values computed with scipy.special / mpmath (double precision).

describe('lgamma', () => {
  it('matches known values', () => {
    expect(lgamma(1)).toBeCloseTo(0, 14);
    expect(lgamma(2)).toBeCloseTo(0, 14);
    expect(lgamma(0.5)).toBeCloseTo(Math.log(Math.sqrt(Math.PI)), 14);
    expect(lgamma(5)).toBeCloseTo(Math.log(24), 13);
    expect(lgamma(10.5)).toBeCloseTo(13.940625219403763, 11);
    expect(lgamma(100)).toBeCloseTo(359.1342053695754, 9);
    expect(lgamma(1e-3)).toBeCloseTo(6.907178885383853, 11);
  });

  it('handles reflection for negative non-integers', () => {
    // Gamma(-0.5) = -2*sqrt(pi), so ln|Gamma| = ln(2 sqrt(pi))
    expect(lgamma(-0.5)).toBeCloseTo(Math.log(2 * Math.sqrt(Math.PI)), 12);
  });

  it('is +Infinity at the poles', () => {
    expect(lgamma(0)).toBe(Infinity);
    expect(lgamma(-3)).toBe(Infinity);
  });
});

describe('digamma', () => {
  it('matches known values', () => {
    const EULER = 0.5772156649015329;
    expect(digamma(1)).toBeCloseTo(-EULER, 12);
    expect(digamma(2)).toBeCloseTo(1 - EULER, 12);
    expect(digamma(0.5)).toBeCloseTo(-EULER - 2 * Math.log(2), 12);
    expect(digamma(10)).toBeCloseTo(2.251752589066721, 12);
    expect(digamma(0.1)).toBeCloseTo(-10.423754940411076, 10);
  });

  it('agrees with the derivative of lgamma', () => {
    for (const x of [0.3, 1.7, 4.2, 25]) {
      const h = 1e-6;
      const fd = (lgamma(x + h) - lgamma(x - h)) / (2 * h);
      expect(digamma(x)).toBeCloseTo(fd, 5);
    }
  });
});

describe('gammainc / gammaincc / gammaincInv', () => {
  it('matches scipy.special.gammainc spot values', () => {
    expect(gammainc(1, 1)).toBeCloseTo(1 - Math.exp(-1), 13);
    expect(gammainc(0.5, 0.5)).toBeCloseTo(0.6826894921370859, 12); // = erf(1/sqrt2)... 2*Phi(1)-1
    expect(gammainc(3, 2)).toBeCloseTo(0.3233235838169365, 12);
    expect(gammainc(10, 10)).toBeCloseTo(0.5420702855281476, 12);
    expect(gammainc(100, 90)).toBeCloseTo(0.15822098918643016, 10);
  });

  it('P + Q = 1 without cancellation', () => {
    for (const [a, x] of [[0.5, 3], [2, 0.1], [20, 30], [5, 5]]) {
      expect(gammainc(a, x) + gammaincc(a, x)).toBeCloseTo(1, 13);
    }
  });

  it('inverse round-trips to 1e-10', () => {
    for (const a of [0.3, 0.5, 1, 2.5, 10, 100]) {
      for (const p of [1e-6, 0.01, 0.3, 0.5, 0.9, 0.999]) {
        const x = gammaincInv(p, a);
        expect(gammainc(a, x)).toBeCloseTo(p, 10);
      }
    }
  });

  it('edge cases', () => {
    expect(gammainc(2, 0)).toBe(0);
    expect(gammainc(2, Infinity)).toBe(1);
    expect(gammaincInv(0, 3)).toBe(0);
    expect(gammaincInv(1, 3)).toBe(Infinity);
  });
});

describe('betainc / betaincInv', () => {
  it('matches scipy.special.betainc spot values', () => {
    expect(betainc(1, 1, 0.3)).toBeCloseTo(0.3, 13); // uniform
    expect(betainc(2, 3, 0.4)).toBeCloseTo(0.5248, 12);
    expect(betainc(0.5, 0.5, 0.5)).toBeCloseTo(0.5, 12); // arcsine symmetric
    expect(betainc(5, 2, 0.8)).toBeCloseTo(0.65536, 11);
    expect(betainc(10, 10, 0.5)).toBeCloseTo(0.5, 12);
  });

  it('respects the symmetry I_x(a,b) = 1 - I_{1-x}(b,a)', () => {
    for (const [a, b, x] of [[2, 5, 0.2], [0.5, 3, 0.7], [8, 1.5, 0.9]]) {
      expect(betainc(a, b, x)).toBeCloseTo(1 - betainc(b, a, 1 - x), 12);
    }
  });

  it('inverse round-trips to 1e-9', () => {
    for (const [a, b] of [[0.5, 0.5], [1, 1], [2, 5], [10, 3], [50, 50]]) {
      for (const p of [1e-5, 0.05, 0.4, 0.5, 0.95, 0.9999]) {
        const x = betaincInv(p, a, b);
        expect(betainc(a, b, x)).toBeCloseTo(p, 9);
      }
    }
  });
});

describe('erf / erfc / normalCdf / normalQuantile', () => {
  it('erf matches known values', () => {
    expect(erf(0)).toBe(0);
    expect(erf(1)).toBeCloseTo(0.8427007929497149, 13);
    expect(erf(-1)).toBeCloseTo(-0.8427007929497149, 13);
    expect(erf(2)).toBeCloseTo(0.9953222650189527, 13);
    expect(erf(0.1)).toBeCloseTo(0.1124629160182849, 13);
  });

  it('erfc avoids cancellation in the far tail', () => {
    expect(erfc(5)).toBeCloseTo(1.5374597944280349e-12, 20);
    expect(erfc(10)).toBeCloseTo(2.0884875837625447e-45, 50);
  });

  it('normalCdf matches known values', () => {
    expect(normalCdf(0)).toBeCloseTo(0.5, 14);
    expect(normalCdf(1.959963984540054)).toBeCloseTo(0.975, 12);
    expect(normalCdf(-3)).toBeCloseTo(0.0013498980316300933, 13);
  });

  it('normalQuantile is the exact inverse to machine precision', () => {
    for (const p of [1e-10, 1e-4, 0.025, 0.31, 0.5, 0.77, 0.975, 1 - 1e-6]) {
      expect(normalCdf(normalQuantile(p))).toBeCloseTo(p, 13);
    }
    expect(normalQuantile(0.975)).toBeCloseTo(1.959963984540054, 12);
    expect(normalQuantile(0.5)).toBeCloseTo(0, 14);
  });

  it('quantile edges', () => {
    expect(normalQuantile(0)).toBe(-Infinity);
    expect(normalQuantile(1)).toBe(Infinity);
  });
});

describe('lbeta / lchoose', () => {
  it('lbeta matches ln B(a,b)', () => {
    expect(lbeta(1, 1)).toBeCloseTo(0, 14);
    expect(lbeta(2, 3)).toBeCloseTo(Math.log(1 / 12), 13);
    expect(lbeta(0.5, 0.5)).toBeCloseTo(Math.log(Math.PI), 13);
  });

  it('lchoose matches exact binomial coefficients', () => {
    expect(Math.exp(lchoose(10, 3))).toBeCloseTo(120, 10);
    expect(Math.exp(lchoose(52, 5))).toBeCloseTo(2598960, 6);
    expect(lchoose(5, 7)).toBe(-Infinity);
  });
});
