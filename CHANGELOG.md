# Changelog

Notable changes to `@tangent.to/proba`. This file starts at 0.2.0; for earlier
releases see the [git history](https://github.com/tangent-to/proba/commits/main)
and the [release tags](https://github.com/tangent-to/proba/releases).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Fixed

- **grad is pinned to 0.3.1 exactly.** A CDN such as jsdelivr bundles each
  release once, with the dependency versions that were newest at that moment.
  With a `^0.3.0` range, proba 0.2.1 was bundled with grad 0.3.0 and mc 0.12.0
  with grad 0.3.1, so a browser loaded two copies of grad. proba then read mc's
  Vars as constants and every observed likelihood dropped out of the model.
  proba and mc now pin the same grad version and are released together.

### Added
- **`logDensity(x, params)` on every distribution**: the log density as a
  `@tangent.to/grad` expression, elementwise, with `x` and any parameter
  allowed to be a `Var`. One function per distribution in `src/density.js`.
  Inside the support it agrees with `logpdf` on plain numbers, and its
  gradients through grad agree with `dlogpdf` in `x` and in every parameter;
  both are tested. mc's observation models and nn's likelihood losses derive
  from it instead of carrying their own copies.

### Changed
- proba now depends on `@tangent.to/grad` (which depends on
  `@tangent.to/lina`). The browser bundle leaves the suite's packages
  external, as mc's does, so a CDN user loads grad alongside proba.
