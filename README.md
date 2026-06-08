# constraint-hamiltonian

[![crates.io](https://img.shields.io/crates/v/constraint-hamiltonian.svg)](https://crates.io/crates/constraint-hamiltonian)
[![docs.rs](https://docs.rs/constraint-hamiltonian/badge.svg)](https://docs.rs/constraint-hamiltonian)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Hamiltonian dynamical systems with holonomic constraints and symplectic integration.**

This crate simulates Hamiltonian systems `H(q, p) = p^T M^{-1} p / 2 + V(q)`
using **symplectic integrators** (Verlet, Störmer-Verlet) that preserve the
symplectic structure — meaning energy drift stays bounded even over long
simulations. Holonomic constraints `g(q) = 0` are projected using SHAKE/RATTLE-
style algorithms.

## Features

- **Hamiltonian systems** — `HamiltonianSystem` with arbitrary potential functions,
  configurable mass matrices, and initial conditions via builder pattern
- **Symplectic integrators** — `SymplecticIntegrator` with `Verlet` and
  `StormerVerlet` methods that preserve phase-space volume
- **Holonomic constraints** — `Constraint` projects trajectories onto the
  constraint manifold using iterative SHAKE/RATTLE projection
- **Energy conservation** — `check_energy_conservation()` and `DriftReport` for
  monitoring energy drift across the integration trajectory
- **Phase portraits** — `PhasePortrait` and `PhasePoint` for visualizing
  trajectories in (q, p) space
- **MIDI export** — `phase_to_midi()` maps phase-space trajectories to MIDI
  events for sonification of dynamical systems

## Quick Start

```rust
use constraint_hamiltonian::{
    HamiltonianSystem, SymplecticIntegrator, IntegrationMethod,
};

// Simple harmonic oscillator: H = p²/2 + q²/2
let mut system = HamiltonianSystem::new(
    vec![1.0],                                // initial position
    vec![0.0],                                // initial momentum
    vec![1.0],                                // mass
    Box::new(|q| 0.5 * q[0] * q[0]),         // potential V(q) = q²/2
    vec![],                                    // no constraints
    0.01,                                      // time step
).unwrap();

let mut integrator = SymplecticIntegrator::new(IntegrationMethod::Verlet, 1000);
let portrait = integrator.integrate(&mut system);

// Verlet preserves energy — drift should be tiny
println!("Max energy drift: {:.6}", integrator.max_energy_drift());
```

## With Constraints

```rust
use constraint_hamiltonian::{HamiltonianSystemBuilder, Constraint};

let system = HamiltonianSystemBuilder::new()
    .position(vec![1.0, 0.0])
    .momentum(vec![0.0, 1.0])
    .mass(vec![1.0, 1.0])
    .potential(Box::new(|q| 0.5 * (q[0]*q[0] + q[1]*q[1])))
    .constraint(Constraint::distance(0, 1, 1.0))  // |q₀ - q₁| = 1
    .dt(0.005)
    .build()
    .unwrap();
```

## Module Overview

| Module | Description |
|---|---|
| `hamiltonian` | `HamiltonianSystem`, `HamiltonianSystemBuilder` — system definition |
| `integrator` | `SymplecticIntegrator`, `IntegrationMethod` — Verlet/Störmer-Verlet |
| `constraint` | `Constraint` — holonomic constraint projection |
| `conservation` | `check_energy_conservation()`, `DriftReport` — energy monitoring |
| `phase` | `PhasePortrait`, `PhasePoint` — trajectory visualization |
| `midi` | `MidiEvent`, `phase_to_midi()`, `export_midi()` — sonification |
| `error` | `HamiltonianError` — error types |

## Links

- [Documentation](https://docs.rs/constraint-hamiltonian)
- [Repository](https://github.com/nightshift-crates/constraint-hamiltonian)
- [Crates.io](https://crates.io/crates/constraint-hamiltonian)

## License

MIT
