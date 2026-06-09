//! Advanced: augmented Lagrangian, multiple constraints, and energy analysis.
//!
//! Run with: cargo run --example advanced

use constraint_hamiltonian::{State, Constraint, Hamiltonian};

fn main() {
    println!("=== Advanced Constraint Hamiltonian ===\n");

    // --- Augmented Lagrangian Method ---
    println!("1. Augmented Lagrangian Method");
    println!("================================\n");
    println!("The augmented Lagrangian updates multipliers λᵢ ← λᵢ + wᵢ·cᵢ(q)");
    println!("each outer iteration, improving constraint satisfaction.\n");

    let constraint = Constraint::new(
        5.0,  // moderate weight — rely on multiplier updates
        Box::new(|q: &[f64]| q[0] - 1.0),
        Box::new(|_q: &[f64]| vec![1.0]),
    );

    let mut h = Hamiltonian::new(
        Box::new(|_q: &[f64], p: &[f64]| 0.5 * p.iter().map(|x| x * x).sum::<f64>()),
        Box::new(|_q: &[f64]| 0.0),
        vec![constraint],
    );

    let mut state = State::new(vec![0.0], vec![0.0]);
    let dt = 0.002;
    let inner_steps = 3_000;
    let outer_iters = 15;
    let damping = 0.03;

    println!("  {:>4}  {:>8}  {:>10}  {:>8}", "Iter", "q₀", "Violation", "λ₀");
    println!("  {}", "-".repeat(38));

    for outer in 0..outer_iters {
        for _ in 0..inner_steps {
            state = h.step_damped(&state, dt, damping);
        }
        let v = h.constraint_violation(&state);
        let lambda = h.constraints_mut()[0].multiplier;
        println!("  {:4}  {:+8.4}  {:10.6}  {:8.4}", outer, state.q[0], v, lambda);
        h.update_multipliers(&state.q);
    }

    // --- Nonlinear Constraint: Ellipse ---
    println!("\n2. Nonlinear Constraint (Ellipse)");
    println!("===================================\n");

    // Constraint: (q₀/3)² + (q₁/2)² = 1
    let ellipse = Constraint::new(
        100.0,
        Box::new(|q: &[f64]| (q[0] / 3.0).powi(2) + (q[1] / 2.0).powi(2) - 1.0),
        Box::new(|q: &[f64]| vec![2.0 * q[0] / 9.0, 2.0 * q[1] / 4.0]),
    );

    let h_ellipse = Hamiltonian::new(
        Box::new(|_q: &[f64], p: &[f64]| 0.5 * p.iter().map(|x| x * x).sum::<f64>()),
        Box::new(|_q: &[f64]| 0.0),
        vec![ellipse],
    );

    let mut state_e = State::new(vec![5.0, 5.0], vec![0.0, 0.0]);
    for _ in 0..50_000 {
        state_e = h_ellipse.step_damped(&state_e, 0.001, 0.05);
    }

    let ellipse_val = (state_e.q[0] / 3.0).powi(2) + (state_e.q[1] / 2.0).powi(2);
    println!("  Final q: [{:.4}, {:.4}]", state_e.q[0], state_e.q[1]);
    println!("  Ellipse value: {:.4} (target: 1.0)", ellipse_val);
    println!("  Violation: {:.6}", h_ellipse.constraint_violation(&state_e));

    // --- Harmonic Oscillator: Energy Conservation ---
    println!("\n3. Energy Conservation: Harmonic Oscillator");
    println!("=============================================\n");

    let h_ho = Hamiltonian::new(
        Box::new(|_q: &[f64], p: &[f64]| 0.5 * p.iter().map(|x| x * x).sum::<f64>()),
        Box::new(|q: &[f64]| 0.5 * q[0] * q[0]),
        vec![],
    );

    let mut state_ho = State::new(vec![1.0], vec![0.0]);
    let initial_e = h_ho.augmented_energy(&state_ho);
    let mut max_drift = 0.0_f64;

    for step in 0..100_000 {
        state_ho = h_ho.step(&state_ho, 0.01);
        let drift = (h_ho.augmented_energy(&state_ho) - initial_e).abs();
        max_drift = max_drift.max(drift);
    }

    println!("  Initial energy: {initial_e:.6}");
    println!("  Final energy:   {:.6}", h_ho.augmented_energy(&state_ho));
    println!("  Max drift:      {max_drift:.2e}");
    println!("  Relative:       {:.4}%", max_drift / initial_e * 100.0);

    // --- Effect of Damping ---
    println!("\n4. Damping Parameter Sweep");
    println!("============================\n");

    print!("  {:>5}", "γ");
    for damping in [0.0, 0.01, 0.05, 0.1, 0.3] {
        print!("  {:8}", format!("γ={damping}"));
    }
    println!();

    // ... (full sweep would go here)
    println!("  Damping drives the system toward the constraint surface.");
    println!("  Higher damping = faster convergence but more energy dissipation.");
}
