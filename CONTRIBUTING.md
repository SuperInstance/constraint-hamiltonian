# Contributing to constraint-hamiltonian

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
git clone https://github.com/SuperInstance/constraint-hamiltonian
cd constraint-hamiltonian
cargo build
cargo test
```

## Architecture

The crate has a flat structure centered on two core types:

- **`State`** — Position `q` and momentum `p` in ℝ²ⁿ
- **`Constraint`** — Constraint function c(q) = 0 with gradient ∇c(q), weight, and multiplier
- **`Hamiltonian`** — Kinetic + potential + constraint penalty, with integrator methods

The integration scheme is **Störmer-Verlet** (symplectic leapfrog):
```
p_{½} = pₙ + (dt/2) F(qₙ)
q_{n+1} = qₙ + dt · p_{½}
p_{n+1} = p_{½} + (dt/2) F(q_{n+1})
```

## Making Changes

### Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy` and fix all warnings
- All public items must have doc comments
- Use trait objects (`Box<dyn Fn>`) for user-supplied functions

### Tests

Every PR must:
1. Pass `cargo test` (all existing tests)
2. Add tests for new functionality
3. Maintain energy conservation guarantee (O(dt²) drift)

### Key Invariants

- Störmer-Verlet must conserve the augmented Hamiltonian to O(dt²)
- Constraint violation must decrease under augmented Lagrangian updates
- Damped steps must not increase constraint violation (monotonically convergent)

## Adding New Features

### New Integration Schemes

Add methods to `Hamiltonian`. Options:
- **Velocity Verlet** (equivalent to Störmer-Verlet)
- **4th-order Yoshida** (higher accuracy, still symplectic)
- **RATTLE** (exact constraint enforcement per step)

### New Constraint Types

The `Constraint` struct is general — any function c(q) = 0 with gradient works. For specialized constraints:

```rust
impl Constraint {
    pub fn equality(target: f64, weight: f64, dim: usize) -> Self { ... }
    pub fn inequality(target: f64, weight: f64, dim: usize) -> Self { ... }
}
```

### New Energy Functions

Add constructors for common potentials:
- Harmonic: V = ½ k q²
- Lennard-Jones: V = 4ε[(σ/r)¹² - (σ/r)⁶]
- Morse: V = D(1 - e^{-α(r-r₀)})²

## Release Checklist

- [ ] `cargo test` passes
- [ ] `cargo clippy` clean
- [ ] `cargo fmt` applied
- [ ] README.md updated if API changed

## Questions?

Open an issue at https://github.com/SuperInstance/constraint-hamiltonian/issues
