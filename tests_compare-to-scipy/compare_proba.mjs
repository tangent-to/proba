#!/usr/bin/env node
/**
 * Helper: evaluate a proba distribution on grids for comparison with
 * scipy.stats. Reads a JSON spec, prints a JSON result.
 *
 * Spec: { dist, params, x: [...], p: [...] }
 * Result: { logpdf: [...], cdf: [...], quantile: [...], mean, variance }
 */

import { readFileSync } from 'node:fs';
import { distributions } from '../src/index.js';

const spec = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const d = distributions[spec.dist];
if (!d) {
  console.error(`unknown distribution: ${spec.dist}`);
  process.exit(1);
}

const out = {
  logpdf: spec.x.map((x) => d.logpdf(x, spec.params)),
  cdf: spec.x.map((x) => d.cdf(x, spec.params)),
  quantile: spec.p.map((p) => d.quantile(p, spec.params)),
  mean: d.mean(spec.params),
  variance: d.variance(spec.params),
};

process.stdout.write(JSON.stringify(out));
