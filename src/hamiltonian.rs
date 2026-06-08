use crate::constraint::{project_momenta, project_positions, Constraint};
use crate::error::{HamiltonianError, Result};

/// A Hamiltonian system: H(q,p) = p^T M^{-1} p / 2 + V(q).
#[allow(clippy::type_complexity)]
pub struct HamiltonianSystem {
    /// Generalized positions q.
    pub positions: Vec<f64>,
    /// Generalized momenta p.
    pub momenta: Vec<f64>,
    /// Diagonal mass matrix entries.
    pub mass_matrix: Vec<f64>,
    /// Potential energy function V(q).
    pub potential: Box<dyn Fn(&[f64]) -> f64>,
    /// Constraints g_i(q) = 0.
    pub constraints: Vec<Constraint>,
    /// Time step.
    pub dt: f64,
}

impl std::fmt::Debug for HamiltonianSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HamiltonianSystem")
            .field("positions", &self.positions)
            .field("momenta", &self.momenta)
            .field("mass_matrix", &self.mass_matrix)
            .field("num_constraints", &self.constraints.len())
            .field("dt", &self.dt)
            .finish()
    }
}

impl HamiltonianSystem {
    /// Create a new Hamiltonian system, validating dimensions.
    #[allow(clippy::type_complexity)]
    pub fn new(
        positions: Vec<f64>,
        momenta: Vec<f64>,
        mass_matrix: Vec<f64>,
        potential: Box<dyn Fn(&[f64]) -> f64>,
        constraints: Vec<Constraint>,
        dt: f64,
    ) -> Result<Self> {
        let n = positions.len();
        if n == 0 {
            return Err(HamiltonianError::EmptySystem);
        }
        if momenta.len() != n {
            return Err(HamiltonianError::DimensionMismatch {
                expected: n,
                found: momenta.len(),
                context: "momenta".into(),
            });
        }
        if mass_matrix.len() != n {
            return Err(HamiltonianError::DimensionMismatch {
                expected: n,
                found: mass_matrix.len(),
                context: "mass_matrix".into(),
            });
        }
        for (i, &m) in mass_matrix.iter().enumerate() {
            if m <= 0.0 {
                return Err(HamiltonianError::NonPositiveMass { index: i, value: m });
            }
        }
        if dt <= 0.0 {
            return Err(HamiltonianError::NonPositiveTimeStep(dt));
        }
        Ok(Self {
            positions,
            momenta,
            mass_matrix,
            potential,
            constraints,
            dt,
        })
    }

    /// Number of degrees of freedom.
    pub fn dof(&self) -> usize {
        self.positions.len()
    }

    /// Kinetic energy: T = p^T M^{-1} p / 2
    pub fn kinetic_energy(&self) -> f64 {
        self.momenta
            .iter()
            .zip(self.mass_matrix.iter())
            .map(|(&p, &m)| p * p / (2.0 * m))
            .sum()
    }

    /// Potential energy: V(q)
    pub fn potential_energy(&self) -> f64 {
        (self.potential)(&self.positions)
    }

    /// Total Hamiltonian energy: H = T + V
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    /// Compute forces: F_i = -∂V/∂q_i using central finite differences.
    pub fn forces(&self) -> Vec<f64> {
        let n = self.positions.len();
        let eps = 1e-8;
        let mut f = vec![0.0; n];
        for i in 0..n {
            let mut q_plus = self.positions.clone();
            let mut q_minus = self.positions.clone();
            q_plus[i] += eps;
            q_minus[i] -= eps;
            f[i] = -((self.potential)(&q_plus) - (self.potential)(&q_minus)) / (2.0 * eps);
        }
        f
    }

    /// Evaluate all constraints, returning (name, value) pairs.
    pub fn evaluate_constraints(&self) -> Vec<(String, f64)> {
        self.constraints
            .iter()
            .map(|c| (c.name.clone(), c.evaluate(&self.positions)))
            .collect()
    }

    /// Check all constraints are satisfied within tolerance.
    pub fn check_constraints(&self) -> std::result::Result<(), Vec<(String, f64, f64)>> {
        let violations: Vec<_> = self
            .constraints
            .iter()
            .filter_map(|c| {
                let val = c.evaluate(&self.positions);
                if val.abs() > c.tolerance {
                    Some((c.name.clone(), val, c.tolerance))
                } else {
                    None
                }
            })
            .collect();
        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }

    /// Project momenta onto the constraint tangent space.
    pub fn project_momenta(&mut self) {
        project_momenta(&self.positions, &mut self.momenta, &self.constraints);
    }

    /// Project positions back onto the constraint manifold.
    pub fn project_positions(&mut self, max_iterations: usize) {
        project_positions(&mut self.positions, &self.constraints, max_iterations);
    }

    /// Create a builder for convenient system construction.
    pub fn builder() -> HamiltonianSystemBuilder {
        HamiltonianSystemBuilder::default()
    }
}

/// Builder for HamiltonianSystem.
#[derive(Default)]
#[allow(clippy::type_complexity)]
pub struct HamiltonianSystemBuilder {
    positions: Option<Vec<f64>>,
    momenta: Option<Vec<f64>>,
    mass_matrix: Option<Vec<f64>>,
    potential: Option<Box<dyn Fn(&[f64]) -> f64>>,
    constraints: Vec<Constraint>,
    dt: Option<f64>,
}

impl HamiltonianSystemBuilder {
    pub fn positions(mut self, positions: Vec<f64>) -> Self {
        self.positions = Some(positions);
        self
    }

    pub fn momenta(mut self, momenta: Vec<f64>) -> Self {
        self.momenta = Some(momenta);
        self
    }

    pub fn mass_matrix(mut self, mass_matrix: Vec<f64>) -> Self {
        self.mass_matrix = Some(mass_matrix);
        self
    }

    pub fn uniform_mass(mut self, n: usize, m: f64) -> Self {
        self.mass_matrix = Some(vec![m; n]);
        self
    }

    #[allow(clippy::type_complexity)]
    pub fn potential(mut self, potential: Box<dyn Fn(&[f64]) -> f64>) -> Self {
        self.potential = Some(potential);
        self
    }

    pub fn zero_potential(mut self) -> Self {
        self.potential = Some(Box::new(|_| 0.0));
        self
    }

    pub fn constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    pub fn dt(mut self, dt: f64) -> Self {
        self.dt = Some(dt);
        self
    }

    pub fn build(self) -> Result<HamiltonianSystem> {
        let positions = self.positions.ok_or_else(|| {
            HamiltonianError::DimensionMismatch {
                expected: 1,
                found: 0,
                context: "positions not set".into(),
            }
        })?;
        let n = positions.len();
        let momenta = self.momenta.unwrap_or_else(|| vec![0.0; n]);
        let mass_matrix = self.mass_matrix.unwrap_or_else(|| vec![1.0; n]);
        let potential = self
            .potential
            .unwrap_or_else(|| Box::new(|_| 0.0));
        let dt = self.dt.unwrap_or(0.01);

        HamiltonianSystem::new(positions, momenta, mass_matrix, potential, self.constraints, dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harmonic_potential() -> Box<dyn Fn(&[f64]) -> f64> {
        Box::new(|q| 0.5 * q.iter().map(|x| x * x).sum::<f64>())
    }

    #[test]
    fn test_system_creation() {
        let sys = HamiltonianSystem::new(
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            harmonic_potential(),
            vec![],
            0.01,
        )
        .unwrap();
        assert_eq!(sys.dof(), 2);
    }

    #[test]
    fn test_kinetic_energy() {
        let sys = HamiltonianSystem::new(
            vec![0.0],
            vec![2.0],
            vec![1.0],
            Box::new(|_| 0.0),
            vec![],
            0.01,
        )
        .unwrap();
        assert!((sys.kinetic_energy() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_potential_energy() {
        let sys = HamiltonianSystem::new(
            vec![3.0],
            vec![0.0],
            vec![1.0],
            Box::new(|q| 0.5 * q[0] * q[0]),
            vec![],
            0.01,
        )
        .unwrap();
        assert!((sys.potential_energy() - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_total_energy() {
        let sys = HamiltonianSystem::new(
            vec![1.0],
            vec![1.0],
            vec![1.0],
            Box::new(|q| 0.5 * q[0] * q[0]),
            vec![],
            0.01,
        )
        .unwrap();
        // T = 0.5, V = 0.5, H = 1.0
        assert!((sys.total_energy() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_forces_harmonic() {
        let sys = HamiltonianSystem::new(
            vec![2.0],
            vec![0.0],
            vec![1.0],
            Box::new(|q| 0.5 * q[0] * q[0]),
            vec![],
            0.01,
        )
        .unwrap();
        let forces = sys.forces();
        // F = -dV/dq = -q = -2.0
        assert!((forces[0] - (-2.0)).abs() < 1e-5);
    }

    #[test]
    fn test_empty_system_rejected() {
        let result = HamiltonianSystem::new(
            vec![],
            vec![],
            vec![],
            Box::new(|_| 0.0),
            vec![],
            0.01,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_momenta() {
        let result = HamiltonianSystem::new(
            vec![1.0],
            vec![1.0, 2.0],
            vec![1.0],
            Box::new(|_| 0.0),
            vec![],
            0.01,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_non_positive_mass() {
        let result = HamiltonianSystem::new(
            vec![1.0],
            vec![0.0],
            vec![0.0],
            Box::new(|_| 0.0),
            vec![],
            0.01,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_dt() {
        let result = HamiltonianSystem::new(
            vec![1.0],
            vec![0.0],
            vec![1.0],
            Box::new(|_| 0.0),
            vec![],
            -0.01,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let sys = HamiltonianSystem::builder()
            .positions(vec![1.0])
            .momenta(vec![0.0])
            .uniform_mass(1, 2.0)
            .potential(Box::new(|q| 0.5 * q[0] * q[0]))
            .dt(0.001)
            .build()
            .unwrap();
        assert_eq!(sys.dof(), 1);
        assert_eq!(sys.mass_matrix[0], 2.0);
    }

    #[test]
    fn test_constraint_evaluation() {
        let c = Constraint::new(
            "unit",
            Box::new(|q| q[0] * q[0] - 1.0),
            Box::new(|q| vec![2.0 * q[0]]),
            1e-8,
        );
        let sys = HamiltonianSystem::new(
            vec![1.0],
            vec![0.0],
            vec![1.0],
            Box::new(|_| 0.0),
            vec![c],
            0.01,
        )
        .unwrap();
        let evals = sys.evaluate_constraints();
        assert_eq!(evals.len(), 1);
        assert!(evals[0].1.abs() < 1e-10);
        assert!(sys.check_constraints().is_ok());
    }
}
