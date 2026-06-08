# constraint-hamiltonian

[![crates.io](https://img.shields.io/crates/v/constraint-hamiltonian.svg)](https://crates.io/crates/constraint-hamiltonian)
[![docs.rs](https://docs.rs/constraint-hamiltonian/badge.svg)](https://docs.rs/constraint-hamiltonian)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## The Problem

Hamiltonian mechanics gives you a beautiful framework: the state of a system lives on a symplectic manifold, and time evolution preserves the symplectic form. Energy is automatically conserved — not because you're careful, but because the geometry demands it.

But real systems have constraints. A pendulum's bob is constrained to a circle. A multi-agent system must conserve total fleet energy. A musical system must stay within a key. These constraints are holonomic — they're functions of the coordinates, not the velocities.

Standard integrators (Runge-Kutta, Euler) don't preserve the symplectic structure. After millions of timesteps, the energy drifts and the constraint is violated. The simulation slowly becomes unphysical.

## The Solution

**Symplectic integrators** (Verlet, Störmer-Verlet) preserve the symplectic form by construction. They never exactly conserve energy, but the energy oscillates around the true value with bounded error — forever. No drift.

`constraint-hamiltonian` implements symplectic integrators **with holonomic constraint projection**. After each symplectic step, the state is projected back onto the constraint manifold using SHAKE/RATTLE-style algorithms. The result: both the symplectic structure and the constraints are preserved.

## How It Works

### Define a Hamiltonian system

```rust
use constraint_hamiltonian::{HamiltonianSystem, Constraint};

let mut system = HamiltonianSystem::new(3); // 3 degrees of freedom
system.set_potential(|q: &[f64]| {
    // Harmonic potential: V = ½k·x²
    0.5 * q.iter().map(|x| x * x).sum::<f64>()
});
system.set_gradient(|q: &[f64]| {
    // dV/dx = k·x
    q.iter().map(|x| x).collect()
});
```

### Add constraints

```rust
// Constrain to a sphere: |q|² = R²
system.add_constraint(Constraint::holonomic(
    |q| q.iter().map(|x| x * x).sum::<f64>() - 1.0,
    |q| q.iter().map(|x| 2.0 * x).collect::<Vec<_>>(), // gradient
));
```

### Integrate with a symplectic integrator

```rust
use constraint_hamiltonian::Integrator;

let mut integrator = Integrator::stormer_verlet(0.001);

for step in 0..100_000 {
    integrator.step(&mut system);
    // Energy is bounded, constraint is satisfied to machine precision
}
```

### Verify conservation

```rust
let energy = system.total_energy();
let constraint_error = system.constraint_violation();
println!("Energy: {:.8}", energy);
println!("Constraint error: {:.2e}", constraint_error);
// Both should be tiny, even after 100k steps
```

## The Math

The Störmer-Verlet method splits the Hamiltonian into kinetic (T) and potential (V) parts:

```
q_{n+1/2} = q_n + (h/2) · p_n/m           (drift)
p_{n+1}   = p_n - h · ∇V(q_{n+1/2})       (kick)
q_{n+1}   = q_{n+1/2} + (h/2) · p_{n+1}/m (drift)
```

This is a second-order symplectic integrator — it preserves the symplectic 2-form ω = Σ dpᵢ ∧ dqᵢ exactly. Energy oscillates but doesn't drift.

For constrained systems, after each step, the SHAKE algorithm projects onto the constraint manifold:
1. Compute the constraint violation g(q)
2. Compute the gradient ∇g
3. Project: q ← q - λ·∇g (find λ by Newton iteration)
4. Repeat until |g(q)| < tolerance

## Integrators

| Method | Order | Symplectic | Notes |
|---|---|---|---|
| Symplectic Euler | 1 | Yes | Fast, least accurate |
| Störmer-Verlet | 2 | Yes | Workhorse, time-reversible |
| Custom splitting | varies | Yes | Split Hamiltonian your way |

## Module Map

| Module | What it does |
|---|---|
| `system` | `HamiltonianSystem` — coordinates, momenta, potential, constraints |
| `integrator` | `Integrator` — Symplectic Euler, Störmer-Verlet, custom splittings |
| `constraint` | `Constraint` — holonomic constraints with automatic gradient |
| `projection` | SHAKE/RATTLE constraint projection algorithms |
| `symplectic` | Symplectic form verification (ω preservation check) |
| `energy` | Energy tracking, drift detection |
| `error` | `HamiltonianError` |

## Design Decisions

- **Why not just use RK4?** RK4 is 4th order and very accurate per step, but it's not symplectic. Over long integrations, energy drifts linearly. For physics simulations that run for billions of timesteps, symplectic is the only game in town.
- **SHAKE vs analytic projection**: SHAKE is iterative and works for any constraint. If you know your constraint manifold analytically (sphere, torus, etc.), you can project directly. Both are supported.
- **Why not symplectic RK?** Symplectic Runge-Kutta methods exist (Gauss-Legendre collocation) but they're implicit and expensive. Verlet is explicit, fast, and sufficient for most applications.

## Links

- [Documentation](https://docs.rs/constraint-hamiltonian)
- [Repository](https://github.com/SuperInstance/constraint-hamiltonian)
- [crates.io](https://crates.io/crates/constraint-hamiltonian)
- Hairer, Lubich, Wanner (2006) — *Geometric Numerical Integration*, the definitive reference

## License

MIT
