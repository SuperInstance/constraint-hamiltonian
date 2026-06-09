# constraint-hamiltonian

**Symplectic constraint dynamics — constraints become conservation laws.**

Standard approaches to constrained optimization treat constraints as penalties or projections. This crate does something fundamentally different: it encodes constraints into a Hamiltonian system, then uses **Störmer-Verlet (symplectic leapfrog) integration** to evolve the system. The result is that the **augmented Hamiltonian is conserved by construction**, meaning constraint violations remain bounded forever — not just for one step.

## The Key Insight

Hamiltonian mechanics has a deep structural property: symplectic integration preserves the symplectic 2-form ω = dp ∧ dq. This means energy (the Hamiltonian H = K + V) oscillates with amplitude O(dt²) but **never drifts**. When we encode constraints into the Hamiltonian:

```
H_aug(q, p) = K(p) + V(q) + Σᵢ [½ wᵢ cᵢ(q)² + λᵢ cᵢ(q)]
```

The augmented energy is conserved by the symplectic integrator. **Constraint satisfaction becomes a conservation law**, not just a target.

## When to Use This

- **Constrained optimization** where constraint violations must stay bounded over many iterations
- **Physical simulation** with holonomic constraints (rigid bodies, molecular dynamics)
- **Control systems** where constraints represent safety boundaries
- **Any system** where you want the constraint penalty energy to oscillate, not drift

## Architecture

```
    ┌─────────────────────────────────────┐
    │         Hamiltonian System           │
    │                                      │
    │  H(q,p) = K(p) + V(q) + penalty(q)  │
    │                                      │
    │  K = ½ p^T M⁻¹ p  (kinetic)        │
    │  V = external potential              │
    │  penalty = Σ (½ wᵢ cᵢ² + λᵢ cᵢ)   │
    │                                      │
    │  State = (q, p)  ∈ ℝ²ⁿ             │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────▼───────────────────────┐
    │    Störmer-Verlet Integrator          │
    │                                       │
    │  p_{½} = pₙ + (dt/2) F(qₙ)          │
    │  q_{n+1} = qₙ + dt · p_{½}          │
    │  p_{n+1} = p_{½} + (dt/2) F(q_{n+1}) │
    │                                       │
    │  Symplectic ⟹ H conserved            │
    └──────────────┬───────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    Undamped           Damped
    (H conserved)   (H → min)
                       │
              Augmented Lagrangian
              λᵢ ← λᵢ + wᵢ cᵢ(q)
```

## Quick Start

```rust
use constraint_hamiltonian::{State, Constraint, Hamiltonian};

// Define constraints: q₀² + q₁² = 1 (unit circle)
let unit_circle = Constraint::new(
    100.0,  // penalty weight
    Box::new(|q: &[f64]| q[0]*q[0] + q[1]*q[1] - 1.0),  // c(q) = 0
    Box::new(|q: &[f64]| vec![2.0*q[0], 2.0*q[1]]),      // ∇c(q)
);

let h = Hamiltonian::new(
    Box::new(|_q: &[f64], p: &[f64]| 0.5 * p.iter().map(|pi| pi*pi).sum::<f64>()),  // K = ½|p|²
    Box::new(|_q: &[f64]| 0.0),  // no external potential
    vec![unit_circle],
);

// Start far from constraint surface
let mut state = State::new(vec![2.0, 0.0], vec![0.0, 0.0]);

// Damped evolution drives toward constraint surface
for _ in 0..30_000 {
    state = h.step_damped(&state, 0.001, 0.05);
}

let r = (state.q[0].powi(2) + state.q[1].powi(2)).sqrt();
println!("Final radius: {r:.4} (target: 1.0)");
println!("Violation: {:.6}", h.constraint_violation(&state));
```

## API Walkthrough

### Defining Constraints

```rust
// q₀ = 1 (fixed value)
let fixed = Constraint::new(
    50.0,
    Box::new(|q: &[f64]| q[0] - 1.0),
    Box::new(|_q: &[f64]| vec![1.0]),
);

// q₀ + q₁ = 1 (linear constraint)
let linear = Constraint::new(
    50.0,
    Box::new(|q: &[f64]| q[0] + q[1] - 1.0),
    Box::new(|_q: &[f64]| vec![1.0, 1.0]),
);

// Multiple constraints simultaneously
let h = Hamiltonian::new(kinetic, potential, vec![fixed, linear]);
```

### Integration Methods

```rust
// Symplectic (energy-conserving)
let next = h.step(&state, 0.01);

// Damped (drives toward constraint surface)
let next = h.step_damped(&state, 0.01, 0.05);

// Augmented Lagrangian: update multipliers periodically
let mut h = h; // mutable
h.update_multipliers(&state.q);
```

### Energy Monitoring

```rust
let ke = h.kinetic_energy(&state);
let pe = h.potential_energy(&state);
let penalty = h.constraint_penalty(&state.q);
let total = h.augmented_energy(&state);
let violation = h.constraint_violation(&state);
```

## Performance

- **Per step**: O(n · m) where n = dimensions, m = constraints
- **Numerical gradient**: 2n evaluations per step
- **Energy conservation**: O(dt²) — no drift over millions of steps
- **Constraint convergence**: O(1/(w·steps)) for penalty method

For performance-critical applications, provide analytical potential gradients instead of relying on numerical differentiation.

## Ecosystem

Part of the **SuperInstance** family:
- `hodge-consensus` — Which disputes will resolve
- `persistence-agent` — Which behavioral patterns are signal vs noise
- `cosmic-web` — Fleet architecture as large-scale cosmic structure
- `constraint-hamiltonian` — Constraint dynamics with symplectic guarantees
- `renormalization-agent` — Multi-scale agent behavior analysis
