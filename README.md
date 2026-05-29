# constraint-hamiltonian

Hamiltonian mechanics where the potential energy encodes constraint violations — Störmer-Verlet integration, augmented Lagrangian methods, and symplectic constraint satisfaction.

## What This Gives You

- **Constraint-aware dynamics** — Define constraint surfaces as `c(q) = 0` and the system naturally evolves toward them
- **Symplectic integration** — Störmer-Verlet (leapfrog) preserves energy structure over long timescales
- **Augmented Lagrangian** — Iteratively tightens constraint satisfaction via multiplier updates, not just brute-force penalties
- **Damped relaxation** — Drive arbitrary initial states onto the constraint manifold
- **Energy tracking** — Monitor augmented Hamiltonian `H = K + V + penalty` for conservation quality
- **Zero dependencies** — Pure Rust, `std` only

## Quick Start

```rust
use constraint_hamiltonian::{State, Constraint, Hamiltonian};

// Define a constraint: q₀² + q₁² = 1 (unit circle)
let constraint = Constraint::new(
    100.0,
    Box::new(|q: &[f64]| q[0] * q[0] + q[1] * q[1] - 1.0),
    Box::new(|q: &[f64]| vec![2.0 * q[0], 2.0 * q[1]]),
);

let h = Hamiltonian::new(
    Box::new(|_q: &[f64], p: &[f64]| 0.5 * p.iter().map(|pi| pi * pi).sum::<f64>()), // kinetic
    Box::new(|_q: &[f64]| 0.0), // no external potential
    vec![constraint],
);

// Integrate toward the constraint surface
let mut state = State::new(vec![2.0, 0.0], vec![0.0, 0.0]);
for _ in 0..30_000 {
    state = h.step_damped(&state, 0.001, 0.05);
}

let r = (state.q[0].powi(2) + state.q[1].powi(2)).sqrt();
assert!((r - 1.0).abs() < 0.05); // landed on the unit circle
```

## API Reference

### `State`

```rust
State::new(q: Vec<f64>, p: Vec<f64>) -> State  // position + momentum
state.dim() -> usize                              // dimensionality
```

### `Constraint`

```rust
Constraint::new(weight, value_fn, gradient_fn) -> Constraint
constraint.value(q)        // c(q) — should be 0 on surface
constraint.gradient(q)     // ∇c(q)
constraint.weight           // penalty coefficient
constraint.multiplier       // Lagrange multiplier (updated by solver)
```

### `Hamiltonian`

```rust
Hamiltonian::new(kinetic_fn, potential_fn, constraints) -> Hamiltonian
h.step(&state, dt)                   // symplectic Verlet step
h.step_damped(&state, dt, damping)   // damped step for relaxation
h.update_multipliers(q)              // augmented Lagrangian outer iteration
h.augmented_energy(&state)           // K + V + Σ(½w·c² + λ·c)
h.constraint_violation(&state)       // ||c(q)||
h.constraint_penalty(q)             // Σ(½w·c² + λ·c)
```

## How It Fits

- **[constraint-dsl](https://github.com/SuperInstance/constraint-dsl)** — Declarative YAML pipelines that compile to constraint graphs; Hamiltonian dynamics can solve them
- **[constraint-mux](https://github.com/SuperInstance/constraint-mux)** — Real-time constraint mux with consonance; Hamiltonian provides the physics engine
- **[creative-engine-rust](https://github.com/SuperInstance/creative-engine-rust)** — Lorenz-based creative systems; Hamiltonian adds constrained dynamics
- **[conservation-protocol](https://github.com/SuperInstance/conservation-protocol)** — Spectral fingerprints for agent identity; Hamiltonian dynamics for constraint-aware agent motion

## Testing

5 tests covering energy conservation, single/multi-constraint convergence, augmented Lagrangian iteration, and symplectic Hamiltonian preservation.

```bash
cargo test
```

## Installation

```bash
git clone https://github.com/SuperInstance/constraint-hamiltonian.git
cd constraint-hamiltonian
cargo build
```

Requires Rust 1.70+. No external dependencies.

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance) ecosystem.
