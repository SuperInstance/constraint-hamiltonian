//! Basic constrained dynamics: drive a point onto the unit circle.
//!
//! Run with: cargo run --example basic

use constraint_hamiltonian::{State, Constraint, Hamiltonian};

fn main() {
    // Kinetic energy: K = ½ |p|²
    let kinetic = Box::new(|_q: &[f64], p: &[f64]| {
        0.5 * p.iter().map(|pi| pi * pi).sum::<f64>()
    });

    // No external potential
    let potential = Box::new(|_q: &[f64]| 0.0);

    // Constraint: q₀² + q₁² = 1 (unit circle)
    let constraint = Constraint::new(
        100.0,
        Box::new(|q: &[f64]| q[0] * q[0] + q[1] * q[1] - 1.0),
        Box::new(|q: &[f64]| vec![2.0 * q[0], 2.0 * q[1]]),
    );

    let h = Hamiltonian::new(kinetic, potential, vec![constraint]);

    // Start far from the constraint surface
    let mut state = State::new(vec![2.0, 1.0], vec![0.0, 0.0]);
    println!("Initial: q = {:?}, violation = {:.6}",
        state.q, h.constraint_violation(&state));

    // Evolve with damping to converge to constraint surface
    let dt = 0.001;
    let damping = 0.05;
    for _ in 0..30_000 {
        state = h.step_damped(&state, dt, damping);
    }

    let r = (state.q[0].powi(2) + state.q[1].powi(2)).sqrt();
    println!("\nFinal:   q = [{:.4}, {:.4}]", state.q[0], state.q[1]);
    println!("Radius:  {:.4} (target: 1.0)", r);
    println!("Violation: {:.6}", h.constraint_violation(&state));
    println!("Augmented energy: {:.6}", h.augmented_energy(&state));
}
