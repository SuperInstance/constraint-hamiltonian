//! # Constraint Hamiltonian Dynamics
//!
//! Symplectic integration for constrained systems. Constraints become
//! conservation laws — the augmented Hamiltonian H = K + V + penalty is
//! preserved by the Störmer-Verlet integrator, so constraint violations
//! oscillate with amplitude O(dt²) but never drift.
//!
//! ## Quick Start
//!
//! ```
//! use constraint_hamiltonian::{State, Constraint, Hamiltonian};
//!
//! let constraint = Constraint::new(
//!     100.0,
//!     Box::new(|q: &[f64]| q[0] + q[1] - 1.0),
//!     Box::new(|_q: &[f64]| vec![1.0, 1.0]),
//! );
//!
//! let h = Hamiltonian::new(
//!     Box::new(|_q: &[f64], p: &[f64]| 0.5 * p.iter().map(|x| x*x).sum::<f64>()),
//!     Box::new(|_q: &[f64]| 0.0),
//!     vec![constraint],
//! );
//!
//! let mut state = State::new(vec![2.0, 2.0], vec![0.0, 0.0]);
//! for _ in 0..30_000 {
//!     state = h.step_damped(&state, 0.001, 0.05);
//! }
//! println!("Violation: {:.6}", h.constraint_violation(&state));
//! ```

use std::f64;

/// System state: position q and momentum p in phase space.
///
/// The state vector lives in ℝ²ⁿ where n = q.len() = p.len().
/// Position `q` lives in configuration space; momentum `p` in momentum space.
///
/// # Example
///
/// ```
/// use constraint_hamiltonian::State;
///
/// let state = State::new(vec![1.0, 0.0], vec![0.0, 1.0]);
/// assert_eq!(state.dim(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct State {
    pub q: Vec<f64>,
    pub p: Vec<f64>,
}

impl State {
    pub fn new(q: Vec<f64>, p: Vec<f64>) -> Self {
        assert_eq!(q.len(), p.len(), "position and momentum must have same dimension");
        State { q, p }
    }

    pub fn dim(&self) -> usize {
        self.q.len()
    }
}

/// A constraint on the system: c(q) = 0 on the constraint surface.
///
/// Constraints are enforced via augmented Lagrangian penalty:
/// `penalty = ½ w·c(q)² + λ·c(q)`
///
/// The gradient ∇c(q) is required for computing forces. The Lagrange multiplier
/// λ is updated via the augmented Lagrangian method: `λ ← λ + w·c(q)`.
///
/// # Example
///
/// ```
/// use constraint_hamiltonian::Constraint;
///
/// // Unit circle: q₀² + q₁² = 1
/// let c = Constraint::new(
///     100.0,
///     Box::new(|q: &[f64]| q[0]*q[0] + q[1]*q[1] - 1.0),
///     Box::new(|q: &[f64]| vec![2.0*q[0], 2.0*q[1]]),
/// );
/// assert!((c.value(&[1.0, 0.0])).abs() < 1e-10); // on constraint surface
/// ```
pub struct Constraint {
    /// Weight (penalty coefficient) for this constraint.
    pub weight: f64,
    /// Constraint function c(q) → f64. Should be 0 on the constraint surface.
    pub value_fn: Box<dyn Fn(&[f64]) -> f64>,
    /// Gradient of the constraint function ∇c(q).
    pub gradient_fn: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    /// Current Lagrange multiplier estimate.
    pub multiplier: f64,
    /// Slack variable for relaxation.
    pub slack: f64,
}

impl Constraint {
    /// Create a new constraint with given weight, value function, and gradient function.
    pub fn new(
        weight: f64,
        value_fn: Box<dyn Fn(&[f64]) -> f64>,
        gradient_fn: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    ) -> Self {
        Constraint {
            weight,
            value_fn,
            gradient_fn,
            multiplier: 0.0,
            slack: 0.0,
        }
    }

    /// Evaluate constraint at given position: c(q).
    pub fn value(&self, q: &[f64]) -> f64 {
        (self.value_fn)(q)
    }

    /// Evaluate constraint gradient at given position: ∇c(q).
    pub fn gradient(&self, q: &[f64]) -> Vec<f64> {
        (self.gradient_fn)(q)
    }
}

type KineticFn = Box<dyn Fn(&[f64], &[f64]) -> f64>;
type PotentialFn = Box<dyn Fn(&[f64]) -> f64>;

/// Hamiltonian system with constraints.
///
/// H(q, p) = K(p) + V(q) + constraint_penalty(q)
///
/// Uses Störmer-Verlet (symplectic leapfrog) integration.
pub struct Hamiltonian {
    /// Kinetic energy K(p). For standard systems: K = ½ p^T M^{-1} p.
    kinetic_fn: KineticFn,
    /// External potential energy V(q).
    potential_fn: PotentialFn,
    /// Constraints enforced via penalty / augmented Lagrangian.
    constraints: Vec<Constraint>,
}

impl Hamiltonian {
    /// Create a new Hamiltonian system.
    pub fn new(
        kinetic_fn: KineticFn,
        potential_fn: PotentialFn,
        constraints: Vec<Constraint>,
    ) -> Self {
        Hamiltonian {
            kinetic_fn,
            potential_fn,
            constraints,
        }
    }

    /// Compute total constraint penalty: Σ (½ w_i c_i² + λ_i c_i).
    pub fn constraint_penalty(&self, q: &[f64]) -> f64 {
        let mut penalty = 0.0;
        for c in &self.constraints {
            let val = c.value(q);
            penalty += 0.5 * c.weight * val * val + c.multiplier * val;
        }
        penalty
    }

    /// Compute total constraint violation magnitude: sqrt(Σ c_i(q)²).
    pub fn constraint_violation(&self, state: &State) -> f64 {
        let mut violation = 0.0;
        for c in &self.constraints {
            let val = c.value(&state.q);
            violation += val * val;
        }
        violation.sqrt()
    }

    /// Gradient of constraint penalty w.r.t. q.
    fn constraint_penalty_grad(&self, q: &[f64]) -> Vec<f64> {
        let n = q.len();
        let mut grad = vec![0.0; n];
        for c in &self.constraints {
            let val = c.value(q);
            let cg = c.gradient(q);
            let coeff = c.weight * val + c.multiplier;
            for i in 0..n {
                grad[i] += coeff * cg[i];
            }
        }
        grad
    }

    /// Numerical gradient of external potential.
    fn potential_grad(&self, q: &[f64]) -> Vec<f64> {
        let n = q.len();
        let eps = 1e-8;
        let mut grad = vec![0.0; n];
        for i in 0..n {
            let mut q_plus = q.to_vec();
            let mut q_minus = q.to_vec();
            q_plus[i] += eps;
            q_minus[i] -= eps;
            grad[i] = ((self.potential_fn)(&q_plus) - (self.potential_fn)(&q_minus)) / (2.0 * eps);
        }
        grad
    }

    /// Compute total force (negative gradient of potential + constraint penalty).
    fn force(&self, q: &[f64]) -> Vec<f64> {
        let pg = self.potential_grad(q);
        let cg = self.constraint_penalty_grad(q);
        let n = q.len();
        (0..n).map(|i| -(pg[i] + cg[i])).collect()
    }

    /// Compute kinetic energy.
    pub fn kinetic_energy(&self, state: &State) -> f64 {
        (self.kinetic_fn)(&state.q, &state.p)
    }

    /// Compute external potential energy.
    pub fn potential_energy(&self, state: &State) -> f64 {
        (self.potential_fn)(&state.q)
    }

    /// Compute augmented Hamiltonian: K + V + constraint_penalty.
    pub fn augmented_energy(&self, state: &State) -> f64 {
        self.kinetic_energy(state) + self.potential_energy(state) + self.constraint_penalty(&state.q)
    }

    /// Advance the system by one time step using Störmer-Verlet (leapfrog) integration.
    ///
    /// p_{n+1/2} = p_n + (dt/2) F(q_n)
    /// q_{n+1}   = q_n + dt * p_{n+1/2}       (unit mass M=I)
    /// p_{n+1}   = p_{n+1/2} + (dt/2) F(q_{n+1})
    pub fn step(&self, state: &State, dt: f64) -> State {
        let n = state.dim();

        // Half-step momentum
        let f = self.force(&state.q);
        let p_half: Vec<f64> = (0..n).map(|i| state.p[i] + 0.5 * dt * f[i]).collect();

        // Full-step position (unit mass: dq/dt = p)
        let q_new: Vec<f64> = (0..n).map(|i| state.q[i] + dt * p_half[i]).collect();

        // Half-step momentum at new position
        let f_new = self.force(&q_new);
        let p_new: Vec<f64> = (0..n).map(|i| p_half[i] + 0.5 * dt * f_new[i]).collect();

        State { q: q_new, p: p_new }
    }

    /// Advance with damping (dissipative). Useful for driving system toward constraint surface.
    /// Damping γ ∈ [0, 1): p ← (1 - γ) * p at each step.
    pub fn step_damped(&self, state: &State, dt: f64, damping: f64) -> State {
        let mut next = self.step(state, dt);
        let factor = 1.0 - damping;
        for p in &mut next.p {
            *p *= factor;
        }
        next
    }

    /// Update Lagrange multipliers using augmented Lagrangian method.
    ///
    /// λ_i ← λ_i + μ_i * c_i(q)
    pub fn update_multipliers(&mut self, q: &[f64]) {
        for c in &mut self.constraints {
            let val = c.value(q);
            c.multiplier += c.weight * val;
        }
    }

    /// Get mutable access to constraints.
    pub fn constraints_mut(&mut self) -> &mut Vec<Constraint> {
        &mut self.constraints
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard kinetic energy: K = ½ |p|²
    fn standard_kinetic(_q: &[f64], p: &[f64]) -> f64 {
        0.5 * p.iter().map(|pi| pi * pi).sum::<f64>()
    }

    #[test]
    fn test_unconstrained_harmonic_oscillator_conserves_energy() {
        // V(q) = ½ k q², with k = 1, m = 1
        // H = ½ p² + ½ q² = const
        let potential = Box::new(|q: &[f64]| 0.5 * q[0] * q[0]);

        let h = Hamiltonian::new(
            Box::new(standard_kinetic),
            potential,
            vec![],
        );

        let mut state = State::new(vec![1.0], vec![0.0]);
        let dt = 0.01;
        let steps = 10_000;
        let initial_energy = h.augmented_energy(&state);

        for _ in 0..steps {
            state = h.step(&state, dt);
        }

        let final_energy = h.augmented_energy(&state);
        let energy_drift = (final_energy - initial_energy).abs();
        // Störmer-Verlet: energy oscillates with amplitude ~O(dt²)
        assert!(
            energy_drift < 1e-4,
            "Energy drift too large: {} (initial: {}, final: {})",
            energy_drift, initial_energy, final_energy
        );
    }

    #[test]
    fn test_single_constraint_converges_to_surface() {
        // Constraint: q₀² + q₁² = 1 (unit circle)
        let potential = Box::new(|_q: &[f64]| 0.0);

        let constraint = Constraint::new(
            100.0,
            Box::new(|q: &[f64]| q[0] * q[0] + q[1] * q[1] - 1.0),
            Box::new(|q: &[f64]| vec![2.0 * q[0], 2.0 * q[1]]),
        );

        let h = Hamiltonian::new(
            Box::new(standard_kinetic),
            potential,
            vec![constraint],
        );

        // Start far from unit circle
        let mut state = State::new(vec![2.0, 0.0], vec![0.0, 0.0]);
        let dt = 0.001;
        let damping = 0.05;

        for _ in 0..30_000 {
            state = h.step_damped(&state, dt, damping);
        }

        let violation = h.constraint_violation(&state);
        assert!(
            violation < 0.05,
            "Constraint violation too large: {}",
            violation
        );

        let r = (state.q[0].powi(2) + state.q[1].powi(2)).sqrt();
        assert!(
            (r - 1.0).abs() < 0.05,
            "Radius should be close to 1.0, got {}",
            r
        );
    }

    #[test]
    fn test_multiple_constraints_satisfied_simultaneously() {
        // Constraint 1: q₀ + q₁ = 1
        // Constraint 2: q₀ - q₁ = 0
        // Solution: q₀ = q₁ = 0.5

        let potential = Box::new(|_q: &[f64]| 0.0);

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
            potential,
            vec![c1, c2],
        );

        let mut state = State::new(vec![2.0, 2.0], vec![0.0, 0.0]);
        let dt = 0.001;
        let damping = 0.05;

        for _ in 0..30_000 {
            state = h.step_damped(&state, dt, damping);
        }

        let violation = h.constraint_violation(&state);
        assert!(
            violation < 0.05,
            "Multi-constraint violation too large: {}",
            violation
        );

        assert!(
            (state.q[0] - 0.5).abs() < 0.05,
            "q₀ should be near 0.5, got {}",
            state.q[0]
        );
        assert!(
            (state.q[1] - 0.5).abs() < 0.05,
            "q₁ should be near 0.5, got {}",
            state.q[1]
        );
    }

    #[test]
    fn test_augmented_lagrangian_decreases_violation() {
        // Constraint: q₀ = 1
        let potential = Box::new(|_q: &[f64]| 0.0);

        let constraint = Constraint::new(
            5.0, // moderate weight
            Box::new(|q: &[f64]| q[0] - 1.0),
            Box::new(|_q: &[f64]| vec![1.0]),
        );

        let mut h = Hamiltonian::new(
            Box::new(standard_kinetic),
            potential,
            vec![constraint],
        );

        let mut state = State::new(vec![0.0], vec![0.0]);
        let dt = 0.002;
        let inner_steps = 3_000;
        let outer_iters = 15;
        let damping = 0.03;

        let mut violations = Vec::new();

        for _ in 0..outer_iters {
            for _ in 0..inner_steps {
                state = h.step_damped(&state, dt, damping);
            }
            let v = h.constraint_violation(&state);
            violations.push(v);
            h.update_multipliers(&state.q);
        }

        let first = violations[0];
        let last = *violations.last().unwrap();
        assert!(
            last < first,
            "Augmented Lagrangian should reduce violation: first={}, last={}",
            first, last
        );

        assert!(
            last < 0.05,
            "Final violation should be small: {}",
            last
        );
    }

    #[test]
    fn test_augmented_hamiltonian_conservation() {
        // With constraints, the augmented Hamiltonian should stay bounded
        // over pure (undamped) symplectic integration.

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

        for _ in 0..steps {
            state = h.step(&state, dt);
            let e = h.augmented_energy(&state);
            let dev = (e - initial_energy).abs();
            if dev > max_deviation {
                max_deviation = dev;
            }
        }

        let relative_deviation = max_deviation / initial_energy.abs().max(1e-10);
        assert!(
            relative_deviation < 0.05,
            "Augmented Hamiltonian deviation too large: {} (relative), initial={}, max_dev={}",
            relative_deviation, initial_energy, max_deviation
        );
    }
}
