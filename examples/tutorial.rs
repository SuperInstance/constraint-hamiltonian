//! Tutorial: step-by-step Hamiltonian constraint dynamics.
//!
//! Run with: cargo run --example tutorial

use constraint_hamiltonian::{State, Constraint, Hamiltonian};

fn main() {
    println!("=== Constraint Hamiltonian Tutorial ===\n");

    // Step 1: The problem
    println!("Step 1: Constrained Dynamics");
    println!("=============================");
    println!("We want to find q such that c(q) = 0,");
    println!("but using Hamiltonian mechanics instead of Lagrange multipliers alone.\n");

    // Step 2: Define the system
    println!("Step 2: Define the Hamiltonian");
    println!("===============================");
    println!("H(q,p) = K(p) + V(q) + penalty(q)");
    println!("  K = ½|p|²  (kinetic energy)");
    println!("  V = 0      (no external potential)");
    println!("  penalty = Σ [½ wᵢ cᵢ(q)² + λᵢ cᵢ(q)]\n");

    let kinetic = Box::new(|_q: &[f64], p: &[f64]| 0.5 * p.iter().map(|x| x * x).sum::<f64>());

    // Step 3: Define constraints
    println!("Step 3: Define Constraints");
    println!("===========================");
    println!("Constraint 1: q₀ + q₁ = 1");
    println!("Constraint 2: q₀ - q₁ = 0");
    println!("Expected solution: q₀ = q₁ = 0.5\n");

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

    let potential = Box::new(|_q: &[f64]| 0.0);
    let h = Hamiltonian::new(kinetic, potential, vec![c1, c2]);

    // Step 4: Set initial conditions
    println!("Step 4: Initial Conditions");
    println!("===========================");
    let mut state = State::new(vec![2.0, 2.0], vec![0.0, 0.0]);
    println!("q₀ = {:?}, p₀ = {:?}", state.q, state.p);
    println!("Initial violation: {:.4}", h.constraint_violation(&state));
    println!("Initial energy:    {:.4}", h.augmented_energy(&state));

    // Step 5: Symplectic integration
    println!("\nStep 5: Störmer-Verlet Integration");
    println!("===================================");
    println!("The leapfrog integrator preserves the symplectic structure,");
    println!("so augmented energy oscillates but never drifts.\n");

    let dt = 0.001;
    let damping = 0.05;
    let total_steps = 30_000;
    let log_interval = 5_000;

    for step in 0..=total_steps {
        if step % log_interval == 0 {
            let v = h.constraint_violation(&state);
            let e = h.augmented_energy(&state);
            println!("  Step {:6}: q=[{:.4}, {:.4}], violation={:.6}, energy={:.4}",
                step, state.q[0], state.q[1], v, e);
        }
        state = h.step_damped(&state, dt, damping);
    }

    // Step 6: Verify convergence
    println!("\nStep 6: Convergence Check");
    println!("==========================");
    println!("Final q₀ = {:.4} (target: 0.5)", state.q[0]);
    println!("Final q₁ = {:.4} (target: 0.5)", state.q[1]);
    println!("Final violation: {:.6}", h.constraint_violation(&state));

    // Step 7: Energy conservation (undamped)
    println!("\nStep 7: Energy Conservation (Undamped)");
    println!("=======================================");
    println!("Without damping, the augmented Hamiltonian is conserved.\n");

    let potential_harmonic = Box::new(|q: &[f64]| 0.5 * (q[0].powi(2) + q[1].powi(2)));
    let constraint_h = Constraint::new(
        10.0,
        Box::new(|q: &[f64]| q[0] + q[1] - 1.0),
        Box::new(|_q: &[f64]| vec![1.0, 1.0]),
    );

    let h2 = Hamiltonian::new(
        Box::new(|_q: &[f64], p: &[f64]| 0.5 * p.iter().map(|x| x * x).sum::<f64>()),
        potential_harmonic,
        vec![constraint_h],
    );

    let mut state2 = State::new(vec![0.5, 0.5], vec![0.1, -0.1]);
    let initial_energy = h2.augmented_energy(&state2);
    let mut max_deviation = 0.0_f64;

    for _ in 0..10_000 {
        state2 = h2.step(&state2, 0.001);
        let dev = (h2.augmented_energy(&state2) - initial_energy).abs();
        max_deviation = max_deviation.max(dev);
    }

    println!("Initial energy:  {initial_energy:.6}");
    println!("Max deviation:   {max_deviation:.6}");
    println!("Relative:        {:.4}%", max_deviation / initial_energy.abs() * 100.0);
    println!("\n=== Tutorial Complete ===");
}
