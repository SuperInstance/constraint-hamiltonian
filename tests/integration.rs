//! Integration tests for constraint-hamiltonian

use constraint_hamiltonian::*;

fn standard_kinetic(_q: &[f64], p: &[f64]) -> f64 {
    0.5 * p.iter().map(|pi| pi * pi).sum::<f64>()
}

#[test]
fn test_state_creation() {
    let s = State::new(vec![1.0, 2.0], vec![3.0, 4.0]);
    assert_eq!(s.dim(), 2);
}

#[test]
#[should_panic(expected = "position and momentum must have same dimension")]
fn test_state_dimension_mismatch() {
    State::new(vec![1.0], vec![1.0, 2.0]);
}

#[test]
fn test_free_particle_motion() {
    // No potential, no constraints: p stays constant, q moves linearly
    let h = Hamiltonian::new(
        Box::new(standard_kinetic),
        Box::new(|_| 0.0),
        vec![],
    );
    let state = State::new(vec![0.0], vec![1.0]); // v = 1
    let after = h.step(&state, 1.0);
    // After dt=1 with p=1, q should be ~1
    assert!((after.q[0] - 1.0).abs() < 0.1, "q should be ~1.0, got {}", after.q[0]);
}

#[test]
fn test_constraint_penalty_zero_on_surface() {
    let c = Constraint::new(
        10.0,
        Box::new(|q: &[f64]| q[0] + q[1] - 1.0),
        Box::new(|_| vec![1.0, 1.0]),
    );
    let h = Hamiltonian::new(
        Box::new(standard_kinetic),
        Box::new(|_| 0.0),
        vec![c],
    );
    // On constraint surface: q0 + q1 = 1
    let penalty = h.constraint_penalty(&[0.5, 0.5]);
    assert!(penalty.abs() < 1e-10, "penalty should be zero on surface");
}

#[test]
fn test_constraint_penalty_nonzero_off_surface() {
    let c = Constraint::new(
        10.0,
        Box::new(|q: &[f64]| q[0] - 1.0),
        Box::new(|_| vec![1.0]),
    );
    let h = Hamiltonian::new(
        Box::new(standard_kinetic),
        Box::new(|_| 0.0),
        vec![c],
    );
    let penalty = h.constraint_penalty(&[2.0]);
    // c(2) = 1, penalty = 0.5 * 10 * 1 + 0 * 1 = 5
    assert!((penalty - 5.0).abs() < 1e-10);
}

#[test]
fn test_constraint_violation_measurement() {
    let c = Constraint::new(
        1.0,
        Box::new(|q: &[f64]| q[0] * q[0] + q[1] * q[1] - 1.0),
        Box::new(|q: &[f64]| vec![2.0 * q[0], 2.0 * q[1]]),
    );
    let h = Hamiltonian::new(
        Box::new(standard_kinetic),
        Box::new(|_| 0.0),
        vec![c],
    );
    // On unit circle: violation = 0
    let on_circle = State::new(vec![1.0, 0.0], vec![0.0, 0.0]);
    assert!(h.constraint_violation(&on_circle) < 1e-10);
    // Off circle: violation > 0
    let off = State::new(vec![2.0, 0.0], vec![0.0, 0.0]);
    assert!(h.constraint_violation(&off) > 0.5);
}

#[test]
fn test_augmented_energy_composition() {
    let h = Hamiltonian::new(
        Box::new(standard_kinetic),
        Box::new(|q: &[f64]| 0.5 * q[0] * q[0]),
        vec![],
    );
    let state = State::new(vec![3.0], vec![4.0]);
    let e = h.augmented_energy(&state);
    // K = 0.5 * 16 = 8, V = 0.5 * 9 = 4.5, total = 12.5
    assert!((e - 12.5).abs() < 1e-10);
}

#[test]
fn test_damped_reduces_momentum() {
    let h = Hamiltonian::new(
        Box::new(standard_kinetic),
        Box::new(|_| 0.0),
        vec![],
    );
    let state = State::new(vec![0.0], vec![10.0]);
    let damped = h.step_damped(&state, 0.01, 0.5);
    assert!(damped.p[0].abs() < state.p[0].abs());
}
