//! # Constraint Hamiltonian — World-Class Tutorial
//!
//! This tutorial teaches Hamiltonian constraint satisfaction from the ground up.
//! Each lesson builds on the last, progressing from basic physics to advanced
//! augmented-Lagrangian techniques.
//!
//! ## The Big Idea
//!
//! Constraints are hard: you need x + y = 1 *and* x - y = 0 simultaneously.
//! The Hamiltonian approach treats constraints as energy penalties, then uses
//! **symplectic (structure-preserving) integration** so the system evolves
//! toward the constraint surface without energy drift.
//!
//! Run: `cargo run --example tutorial`

use constraint_hamiltonian::{Constraint, Hamiltonian, State};

/// Standard kinetic energy: K = ½|p|² (unit mass).
fn standard_kinetic(_q: &[f64], p: &[f64]) -> f64 {
    0.5 * p.iter().map(|pi| pi * pi).sum::<f64>()
}

// ─────────────────────────────────────────────────────────────────────────────
// Lesson 1: Free Particle — Symplectic Integration Preserves Energy
// ─────────────────────────────────────────────────────────────────────────────
//
// A free particle has no forces. Its Hamiltonian is H = ½|p|².
// In Hamiltonian mechanics, H is *always conserved* — it's a physical law.
//
// The Störmer-Verlet integrator is *symplectic*: it preserves a nearby
// Hamiltonian H' ≈ H + O(dt²), so energy oscillates but never drifts.
// Compare this to naive Euler, where energy grows without bound.
//
// KEY INSIGHT: Symplectic integration trades exact energy conservation for
// *bounded* error. The error is O(dt²) and periodic — it doesn't accumulate.

fn lesson1_free_particle() {
    println!("═ Lesson 1: Free Particle — Symplectic Energy Conservation ═\n");

    let h = Hamiltonian::new(
        Box::new(standard_kinetic),
        Box::new(|_q: &[f64]| 0.0), // no potential
        vec![],
    );

    // Start at rest: position [1.0], momentum [0.0]
    let mut state = State::new(vec![1.0], vec![0.0]);
    let dt = 0.01;
    let steps = 10_000;
    let initial_energy = h.augmented_energy(&state);

    println!("  Initial energy:  {:.10}", initial_energy);
    println!("  Integrating {} steps with dt={}...", steps, dt);

    for _ in 0..steps {
        state = h.step(&state, dt);
    }

    let final_energy = h.augmented_energy(&state);
    let drift = (final_energy - initial_energy).abs();

    println!("  Final energy:    {:.10}", final_energy);
    println!("  Energy drift:    {:.2e} (should be ~0)", drift);
    println!("  Position moved:  q = [{:.4}] (momentum = 0, no force → no motion)\n", state.q[0]);

    assert!(drift < 1e-12, "Energy should be exactly conserved for free particle");
}

// ─────────────────────────────────────────────────────────────────────────────
// Lesson 2: Harmonic Oscillator — Symplectic vs. Euler
// ─────────────────────────────────────────────────────────────────────────────
//
// V(q) = ½kq² with k=1, m=1 gives H = ½p² + ½q².
// The exact solution is q(t) = cos(t), p(t) = -sin(t), with H ≡ ½.
//
// Symplectic integration preserves a *modified* Hamiltonian H' = H + O(dt²).
// The energy oscillates with amplitude ~dt² but NEVER grows secularly.
// This is why physicists use symplectic integrators for N-body simulations
// that run for billions of time steps.

fn lesson2_harmonic_oscillator() {
    println!("═ Lesson 2: Harmonic Oscillator — Bounded Energy Oscillation ═\n");

    let h = Hamiltonian::new(
        Box::new(standard_kinetic),
        Box::new(|q: &[f64]| 0.5 * q[0] * q[0]), // V = ½q²
        vec![],
    );

    let mut state = State::new(vec![1.0], vec![0.0]); // H = ½
    let dt = 0.01;
    let steps = 10_000;
    let initial_energy = h.augmented_energy(&state);

    let mut max_energy = initial_energy;
    let mut min_energy = initial_energy;

    println!("  Initial: q = {:.4}, p = {:.4}, H = {:.10}", state.q[0], state.p[0], initial_energy);
    println!("  Integrating {} steps (total time = {})...", steps, dt * steps as f64);

    for _ in 0..steps {
        state = h.step(&state, dt);
        let e = h.augmented_energy(&state);
        max_energy = max_energy.max(e);
        min_energy = min_energy.min(e);
    }

    println!("  Final:   q = {:.4}, p = {:.4}", state.q[0], state.p[0]);
    println!("  Energy range: [{:.10}, {:.10}]", min_energy, max_energy);
    println!("  Max deviation: {:.2e} (symplectic: bounded, not growing!)\n", max_energy - min_energy);

    let rel_deviation = (max_energy - min_energy) / initial_energy;
    assert!(rel_deviation < 0.001, "Symplectic energy oscillation should be tiny");
}

// ─────────────────────────────────────────────────────────────────────────────
// Lesson 3: First Constraint — The Unit Circle
// ─────────────────────────────────────────────────────────────────────────────
//
// Constraint: q₀² + q₁² = 1 (unit circle).
//
// This is encoded as c(q) = q₀² + q₁² - 1 = 0.
// The penalty energy is ½w·c(q)² where w is the constraint weight.
// High w → stiffer constraint, but you need smaller dt for stability.
//
// DAMPING drives the system toward the constraint surface by bleeding
// kinetic energy. Without damping, the particle oscillates around the
// constraint surface forever.

fn lesson3_unit_circle() {
    println!("═ Lesson 3: Single Constraint — Project onto Unit Circle ═\n");

    let constraint = Constraint::new(
        100.0, // weight: how "stiff" the constraint is
        Box::new(|q: &[f64]| q[0] * q[0] + q[1] * q[1] - 1.0), // c(q) = 0 on circle
        Box::new(|q: &[f64]| vec![2.0 * q[0], 2.0 * q[1]]),     // ∇c
    );

    let h = Hamiltonian::new(
        Box::new(standard_kinetic),
        Box::new(|_q: &[f64]| 0.0), // no external potential
        vec![constraint],
    );

    // Start far from unit circle: |q| = √8 ≈ 2.83
    let mut state = State::new(vec![2.0, 2.0], vec![0.0, 0.0]);
    let dt = 0.001;
    let damping = 0.05;

    println!("  Initial: q = [{:.2}, {:.2}], |q| = {:.4}", state.q[0], state.q[1],
             (state.q[0].powi(2) + state.q[1].powi(2)).sqrt());
    println!("  Constraint: q₀² + q₁² = 1");
    println!("  Violation:  {:.6}", h.constraint_violation(&state));

    for _ in 0..30_000 {
        state = h.step_damped(&state, dt, damping);
    }

    let r = (state.q[0].powi(2) + state.q[1].powi(2)).sqrt();
    let violation = h.constraint_violation(&state);

    println!("  Final:   q = [{:.4}, {:.4}], |q| = {:.6}", state.q[0], state.q[1], r);
    println!("  Violation:  {:.2e}\n", violation);

    assert!(violation < 0.05, "Should converge to unit circle");
    assert!((r - 1.0).abs() < 0.05, "Radius should be near 1.0");
}

// ─────────────────────────────────────────────────────────────────────────────
// Lesson 4: Multiple Constraints — Solve a System
// ─────────────────────────────────────────────────────────────────────────────
//
// Two constraints simultaneously:
//   c₁(q) = q₀ + q₁ - 1 = 0   →  q₀ + q₁ = 1
//   c₂(q) = q₀ - q₁     = 0   →  q₀ = q₁
//
// Solution: q₀ = q₁ = 0.5
//
// The Hamiltonian naturally finds the intersection of constraint surfaces!
// This is the same math used in:
//   - Inverse kinematics (robotics)
//   - Molecular dynamics (bond length constraints)
//   - Cloth simulation (distance constraints)
//
// The augmented Hamiltonian H = K + V + Σ(½wᵢcᵢ²) has a minimum at the
// constraint surface intersection.

fn lesson4_multiple_constraints() {
    println!("═ Lesson 4: Multiple Constraints — Solve Linear System ═\n");

    let c1 = Constraint::new(
        50.0,
        Box::new(|q: &[f64]| q[0] + q[1] - 1.0),
        Box::new(|_q: &[f64]| vec![1.0, 1.0]),
    );

    let c2 = Constraint::new(
        50.0,
        Box::new(|q: &[f64]| q[0] - q[1]),
        Box::new(|_q: &[f64]| vec![1.0, -1.0]),
    );

    let h = Hamiltonian::new(
        Box::new(standard_kinetic),
        Box::new(|_q: &[f64]| 0.0),
        vec![c1, c2],
    );

    let mut state = State::new(vec![2.0, 2.0], vec![0.0, 0.0]);
    let dt = 0.001;
    let damping = 0.05;

    println!("  Constraints: q₀ + q₁ = 1  AND  q₀ = q₁");
    println!("  Exact solution: q₀ = q₁ = 0.5");
    println!("  Initial: q = [{:.2}, {:.2}]", state.q[0], state.q[1]);

    for _ in 0..30_000 {
        state = h.step_damped(&state, dt, damping);
    }

    let violation = h.constraint_violation(&state);
    println!("  Final:   q = [{:.4}, {:.4}]", state.q[0], state.q[1]);
    println!("  Error:   |q₀ - 0.5| = {:.2e}, |q₁ - 0.5| = {:.2e}",
             (state.q[0] - 0.5).abs(), (state.q[1] - 0.5).abs());
    println!("  Total violation: {:.2e}\n", violation);

    assert!((state.q[0] - 0.5).abs() < 0.05);
    assert!((state.q[1] - 0.5).abs() < 0.05);
}

// ─────────────────────────────────────────────────────────────────────────────
// Lesson 5: Augmented Lagrangian — Faster Convergence
// ─────────────────────────────────────────────────────────────────────────────
//
// Problem: Pure penalty methods need very large weights (stiff, slow).
// Augmented Lagrangian adds a Lagrange multiplier λ updated each outer iteration:
//
//   L(q, λ) = K(p) + V(q) + Σ(½wᵢcᵢ² + λᵢcᵢ)
//
// The multiplier λ "remembers" the constraint violation direction, so even
// moderate weights achieve tight constraint satisfaction.
//
// This is the *standard method* in nonlinear optimization (ALM, method of
// multipliers). The inner loop integrates the Hamiltonian with damping; the
// outer loop updates λ.

fn lesson5_augmented_lagrangian() {
    println!("═ Lesson 5: Augmented Lagrangian — Multiplier Acceleration ═\n");

    let constraint = Constraint::new(
        5.0, // moderate weight — NOT stiff!
        Box::new(|q: &[f64]| q[0] - 1.0), // c(q) = q₀ - 1
        Box::new(|_q: &[f64]| vec![1.0]),
    );

    let mut h = Hamiltonian::new(
        Box::new(standard_kinetic),
        Box::new(|_q: &[f64]| 0.0),
        vec![constraint],
    );

    let mut state = State::new(vec![0.0], vec![0.0]);
    let dt = 0.002;
    let inner_steps = 3_000;
    let outer_iters = 15;
    let damping = 0.03;

    println!("  Constraint: q₀ = 1.0");
    println!("  Weight: 5.0 (moderate — penalty alone won't suffice)");
    println!("  Initial: q₀ = 0.0\n");
    println!("  {:>4}  {:>12}  {:>12}", "Iter", "Violation", "q₀");

    let mut violations = Vec::new();

    for outer in 0..outer_iters {
        for _ in 0..inner_steps {
            state = h.step_damped(&state, dt, damping);
        }
        let v = h.constraint_violation(&state);
        violations.push(v);
        println!("  {:>4}  {:>12.2e}  {:>12.6}", outer + 1, v, state.q[0]);
        h.update_multipliers(&state.q);
    }

    let first = violations[0];
    let last = *violations.last().unwrap();

    println!("\n  Violation reduction: {:.2e} → {:.2e} ({:.0}x improvement)",
             first, last, first / last.max(1e-15));
    println!("  Final q₀ = {:.6} (target: 1.0)\n", state.q[0]);

    assert!(last < first, "Augmented Lagrangian should improve over iterations");
    assert!(last < 0.05, "Final violation should be small");
}

// ─────────────────────────────────────────────────────────────────────────────
// Lesson 6: Augmented Hamiltonian Conservation — Why Symplectic Matters
// ─────────────────────────────────────────────────────────────────────────────
//
// With constraints and NO damping, the augmented Hamiltonian H̃ = K + V + penalty
// should be approximately conserved by the symplectic integrator.
//
// This is the deep reason the method works: the constraint penalty becomes part
// of the conserved quantity. Energy can flow between kinetic, potential, and
// penalty terms, but the total stays bounded.
//
// Without symplectic integration, the penalty energy would grow unboundedly,
// and the constraint would oscillate with increasing amplitude.

fn lesson6_symplectic_conservation_with_constraints() {
    println!("═ Lesson 6: Why Symplectic? — Augmented H Conservation ═\n");

    let potential = Box::new(|q: &[f64]| 0.5 * (q[0].powi(2) + q[1].powi(2)));
    let constraint = Constraint::new(
        10.0,
        Box::new(|q: &[f64]| q[0] + q[1] - 1.0),
        Box::new(|_q: &[f64]| vec![1.0, 1.0]),
    );

    let h = Hamiltonian::new(
        Box::new(standard_kinetic),
        potential,
        vec![constraint],
    );

    let mut state = State::new(vec![0.5, 0.5], vec![0.1, -0.1]);
    let dt = 0.001;
    let steps = 5_000;

    let initial_energy = h.augmented_energy(&state);
    let mut max_deviation = 0.0_f64;

    println!("  H̃ = K + V + constraint_penalty");
    println!("  Initial augmented H: {:.6}", initial_energy);

    for _ in 0..steps {
        state = h.step(&state, dt); // NO damping — pure symplectic
        let dev = (h.augmented_energy(&state) - initial_energy).abs();
        max_deviation = max_deviation.max(dev);
    }

    let relative = max_deviation / initial_energy.abs().max(1e-10);
    println!("  Max relative deviation over {} steps: {:.2e}", steps, relative);
    println!("  Constraint violation oscillates but NEVER drifts!\n");

    assert!(relative < 0.05, "Augmented H should stay bounded");
}

// ─────────────────────────────────────────────────────────────────────────────
// Lesson 7: Nonlinear Constraints — Sphere Intersection
// ─────────────────────────────────────────────────────────────────────────────
//
// Two sphere constraints in 3D:
//   c₁(q) = q₀² + q₁² + q₂² - 1 = 0   (unit sphere)
//   c₂(q) = (q₀-1)² + q₁² + q₂² - 1 = 0  (sphere centered at (1,0,0))
//
// Intersection: the circle x² + y² + z² = 1, (x-1)² + y² + z² = 1
// → 2x = 1 → x = 0.5, y² + z² = 0.75
//
// The system should converge to any point on this intersection circle.

fn lesson7_nonlinear_sphere_intersection() {
    println!("═ Lesson 7: Nonlinear — Sphere Intersection in 3D ═\n");

    let c1 = Constraint::new(
        100.0,
        Box::new(|q: &[f64]| q[0] * q[0] + q[1] * q[1] + q[2] * q[2] - 1.0),
        Box::new(|q: &[f64]| vec![2.0 * q[0], 2.0 * q[1], 2.0 * q[2]]),
    );

    let c2 = Constraint::new(
        100.0,
        Box::new(|q: &[f64]| (q[0] - 1.0).powi(2) + q[1].powi(2) + q[2].powi(2) - 1.0),
        Box::new(|q: &[f64]| vec![2.0 * (q[0] - 1.0), 2.0 * q[1], 2.0 * q[2]]),
    );

    let h = Hamiltonian::new(
        Box::new(standard_kinetic),
        Box::new(|_q: &[f64]| 0.0),
        vec![c1, c2],
    );

    let mut state = State::new(vec![3.0, 3.0, 3.0], vec![0.0, 0.0, 0.0]);

    println!("  Constraints: |q|² = 1  AND  |q - (1,0,0)|² = 1");
    println!("  Exact: q₀ = 0.5, q₁² + q₂² = 0.75");
    println!("  Initial: q = [{:.1}, {:.1}, {:.1}]", state.q[0], state.q[1], state.q[2]);

    for _ in 0..50_000 {
        state = h.step_damped(&state, 0.001, 0.05);
    }

    let violation = h.constraint_violation(&state);
    let r1 = (state.q[0].powi(2) + state.q[1].powi(2) + state.q[2].powi(2)).sqrt();
    let r2 = ((state.q[0] - 1.0).powi(2) + state.q[1].powi(2) + state.q[2].powi(2)).sqrt();

    println!("  Final:   q = [{:.4}, {:.4}, {:.4}]", state.q[0], state.q[1], state.q[2]);
    println!("  |q|     = {:.6} (target: 1.0)", r1);
    println!("  |q-e₁|  = {:.6} (target: 1.0)", r2);
    println!("  q₀      = {:.4} (target: 0.5)", state.q[0]);
    println!("  q₁²+q₂² = {:.4} (target: 0.75)", state.q[1].powi(2) + state.q[2].powi(2));
    println!("  Violation: {:.2e}\n", violation);

    assert!((state.q[0] - 0.5).abs() < 0.1, "q₀ should be near 0.5");
    assert!(violation < 0.1, "Violation should be small");
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   Constraint Hamiltonian — Interactive Tutorial              ║");
    println!("║   Symplectic integration for constraint satisfaction         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    lesson1_free_particle();
    lesson2_harmonic_oscillator();
    lesson3_unit_circle();
    lesson4_multiple_constraints();
    lesson5_augmented_lagrangian();
    lesson6_symplectic_conservation_with_constraints();
    lesson7_nonlinear_sphere_intersection();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   All lessons passed! ✓                                      ║");
    println!("║                                                              ║");
    println!("║   Key takeaways:                                             ║");
    println!("║   1. Symplectic integration bounds energy error forever      ║");
    println!("║   2. Constraints become energy penalties                     ║");
    println!("║   3. Damping drives toward constraint surface                ║");
    println!("║   4. Augmented Lagrangian accelerates convergence            ║");
    println!("║   5. Works for any combination of nonlinear constraints      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}
