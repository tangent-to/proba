/**
 * Seedable pseudo-random number generation: xoshiro128** seeded through
 * splitmix32. Deterministic across platforms, fast, and good enough for
 * statistical simulation (not for cryptography).
 */

/**
 * Create a seedable RNG.
 *
 * @param {number} [seed] - Any finite number; omit for a time-based seed
 * @returns {Object} {float, int, normal, seed}
 */
export function createRng(seed = Date.now()) {
  // splitmix32 to spread the seed into four non-zero 32-bit words
  let sm = seed >>> 0;
  const splitmix = () => {
    sm = (sm + 0x9e3779b9) >>> 0;
    let z = sm;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad);
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97);
    return (z ^ (z >>> 15)) >>> 0;
  };
  let s0 = splitmix();
  let s1 = splitmix();
  let s2 = splitmix();
  let s3 = splitmix();
  if ((s0 | s1 | s2 | s3) === 0) s0 = 1;

  const rotl = (x, k) => ((x << k) | (x >>> (32 - k))) >>> 0;

  /** Next uint32 from xoshiro128** */
  function int() {
    const result = (Math.imul(rotl(Math.imul(s1, 5) >>> 0, 7), 9)) >>> 0;
    const t = (s1 << 9) >>> 0;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = rotl(s3, 11);
    return result;
  }

  /** Uniform float in [0, 1) with 53-bit resolution */
  function float() {
    const hi = int() >>> 6; // 26 bits
    const lo = int() >>> 5; // 27 bits
    return (hi * 134217728 + lo) / 9007199254740992;
  }

  // Standard normal via polar Box-Muller with a spare
  let spare = null;
  function normal() {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u, v, s;
    do {
      u = 2 * float() - 1;
      v = 2 * float() - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const mul = Math.sqrt(-2 * Math.log(s) / s);
    spare = v * mul;
    return u * mul;
  }

  return { float, int, normal, seed };
}
